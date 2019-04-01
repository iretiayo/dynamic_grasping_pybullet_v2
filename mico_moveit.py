import numpy as np
import os

import rospy
import moveit_commander as mc
from moveit_msgs.msg import DisplayTrajectory, PositionIKRequest, RobotState
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionFK

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header


class MicoMoveit(object):
    TIP_LINK = "m1n6s200_end_effector"
    BASE_LINK = "root"
    ARM = "arm"
    GRIPPER = "gripper"
    ARM_JOINT_NAMES = ['m1n6s200_joint_1',
                       'm1n6s200_joint_2',
                       'm1n6s200_joint_3',
                       'm1n6s200_joint_4',
                       'm1n6s200_joint_5',
                       'm1n6s200_joint_6', ]
    GRIPPER_JOINT_NAMES = ["m1n6s200_joint_finger_1", "m1n6s200_joint_finger_2"]
    JOINT_NAMES = ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES

    OPEN_POSITION = [0.0, 0.0]
    CLOSED_POSITION = [1.1, 1.1]
    FINGER_MAXTURN = 1.3

    def __init__(self):
        # the service names have to be this
        self.arm_ik_svr = rospy.ServiceProxy('compute_ik', GetPositionIK)
        self.arm_fk_svr = rospy.ServiceProxy('compute_fk', GetPositionFK)

        try:
            self.arm_commander_group = mc.MoveGroupCommander('arm')
        except RuntimeError as e:
            print("exception caught")
            print("exception: {}".format(e))
            os.system("kill -9 $(pgrep -f trajectory_execution_server.py)")
            os.system("kill -9 $(pgrep -f motion_prediction_server.py)")
            os.system("kill -9 $(pgrep -f demo.py)")
            exit(0)

        # self.arm_commander_group.set_goal_joint_tolerance(0.5)
        # self.arm_commander_group.set_goal_orientation_tolerance(0.5)
        # self.arm_commander_group.set_goal_position_tolerance(0.5)
        self.robot = mc.RobotCommander()
        self.scene = mc.PlanningSceneInterface()
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                            DisplayTrajectory,
                                                            queue_size=20)
        self.eef_link = self.arm_commander_group.get_end_effector_link()

    def plan(self, start_joint_values, goal_joint_values):
        """ No matter what start and goal are, the returned plan start and goal will
            make circular joints within [-pi, pi] """
        # setup moveit_start_state
        start_robot_state = self.robot.get_current_state()
        start_robot_state.joint_state.name = self.ARM_JOINT_NAMES
        start_robot_state.joint_state.position = start_joint_values

        self.arm_commander_group.set_start_state(start_robot_state)
        self.arm_commander_group.set_joint_value_target(goal_joint_values)
        # takes around 0.11 second
        plan = self.arm_commander_group.plan()
        return plan

    def plan_ee_pose(self, start_joint_values, goal_ee_pose):
        """ using set_pose_target instead """
        # setup moveit_start_state
        start_robot_state = self.robot.get_current_state()
        start_robot_state.joint_state.name = self.ARM_JOINT_NAMES
        start_robot_state.joint_state.position = start_joint_values

        self.arm_commander_group.set_start_state(start_robot_state)
        self.arm_commander_group.set_pose_target(goal_ee_pose)

        plan = self.arm_commander_group.plan()
        return plan

    def plan_straight_line(self, start_joint_values, end_eef_pose, ee_step=0.05, jump_threshold=3.0, avoid_collisions=True):
        """
        :param start_joint_values: start joint values
        :param end_eef_pose: goal end effector pose
        :param ee_step: float. The distance in meters to interpolate the path.
        :param jump_threshold: The maximum allowable distance in the arm's
            configuration space allowed between two poses in the path. Used to
            prevent "jumps" in the IK solution.
        :param avoid_collisions: bool. Whether to check for obstacles or not.
        :return:
        """
        # set moveit start state
        joint_state = JointState()
        joint_state.name = self.ARM_JOINT_NAMES
        joint_state.position = start_joint_values
        moveit_robot_state = RobotState()
        moveit_robot_state.joint_state = joint_state
        self.arm_commander_group.set_start_state(moveit_robot_state)

        start_eef_pose = self.get_arm_fk(start_joint_values)
        plan, fraction = self.arm_commander_group.compute_cartesian_path(
            [start_eef_pose, end_eef_pose],
            ee_step,
            jump_threshold,
            avoid_collisions)
        # remove the first redundant point
        plan.joint_trajectory.points = plan.joint_trajectory.points[1:]
        return plan, fraction

    @staticmethod
    def convert_range(joint_values):
        """ Convert continuous joint to range [-pi, pi] """
        circular_idx = [0, 3, 4, 5]
        new_joint_values = []
        for i, v in enumerate(joint_values):
            if v > np.pi and i in circular_idx:
                new_joint_values.append(v - 2 * np.pi)
            elif v < -np.pi and i in circular_idx:
                new_joint_values.append(v + 2 * np.pi)
            else:
                new_joint_values.append(v)
        return new_joint_values

    def get_arm_eff_pose(self):
        """ Return [[x, y, x], [x, y, z, w]]"""
        pose_stamped = self.arm_commander_group.get_current_pose()
        pose = pose_stamped.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return [position, orientation]

    def get_arm_ik(self, pose_2d, timeout, avoid_collisions, arm_joint_values, gripper_joint_values):
        """
        Compute arm IK.
        :param pose_2d: 2d list, [[x, y, z], [x, y, z, w]]
        :param timeout: timeout in seconds
        :param avoid_collisions: whether to avoid collisions when computing ik
        :param arm_joint_values: arm joint values to seed the IK
        :param gripper_joint_values: gripper joint values for computing IK
        :return: a list of joint values if success; None if no ik
        """
        # when there is collision, we need timeout to control the time to search
        rospy.wait_for_service('compute_ik')

        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = self.BASE_LINK
        gripper_pose_stamped.header.stamp = rospy.Time.now()
        gripper_pose_stamped.pose = Pose(Point(*pose_2d[0]), Quaternion(*pose_2d[1]))

        service_request = PositionIKRequest()
        service_request.group_name = self.ARM
        service_request.ik_link_name = self.TIP_LINK
        service_request.pose_stamped = gripper_pose_stamped
        service_request.timeout.secs = timeout
        service_request.avoid_collisions = avoid_collisions

        seed_robot_state = self.robot.get_current_state()
        seed_robot_state.joint_state.name = self.JOINT_NAMES
        seed_robot_state.joint_state.position = arm_joint_values + gripper_joint_values
        service_request.robot_state = seed_robot_state

        try:
            resp = self.arm_ik_svr(ik_request=service_request)
            if resp.error_code.val == -31:
                # print("No ik exists!")
                return None
            elif resp.error_code.val == 1:
                return self.parse_joint_state_arm(resp.solution.joint_state)
            else:
                print("Other errors!")
                return None
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def get_arm_fk(self, arm_joint_values):
        """ return a ros pose """
        rospy.wait_for_service('compute_fk')

        header = Header(frame_id="world")
        fk_link_names = [self.TIP_LINK]
        robot_state = RobotState()
        robot_state.joint_state.name = self.ARM_JOINT_NAMES
        robot_state.joint_state.position = arm_joint_values


        try:
            resp = self.arm_fk_svr(header=header, fk_link_names=fk_link_names, robot_state=robot_state)
            if resp.error_code.val != 1:
                print("error ({}) happens when computing fk".format(resp.error_code.val))
                return None
            else:
                return resp.pose_stamped[0].pose
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def parse_joint_state_arm(self, joint_state):
        d = {n: v for (n, v) in zip(joint_state.name, joint_state.position)}
        return [d[n] for n in self.ARM_JOINT_NAMES]

    ''' scene and collision '''

    def clear_scene(self):
        for obj_name in self.get_attached_object_names():
            self.scene.remove_attached_object(self.eef_link, obj_name)
        for obj_name in self.get_known_object_names():
            self.scene.remove_world_object(obj_name)

    def add_box(self, name, pose_2d, size=(1, 1, 1)):
        pose_stamped = PoseStamped()
        pose = Pose(Point(*pose_2d[0]), Quaternion(*pose_2d[1]))
        pose_stamped.pose = pose
        pose_stamped.header.frame_id = "/world"
        self.scene.add_box(name, pose_stamped, size)

    def add_mesh(self, name, pose_2d, mesh_filepath, size=(1, 1, 1)):
        pose_stamped = PoseStamped()
        pose = Pose(Point(*pose_2d[0]), Quaternion(*pose_2d[1]))
        pose_stamped.pose = pose
        pose_stamped.header.frame_id = "/world"
        self.scene.add_mesh(name, pose_stamped, mesh_filepath, size)

    def get_known_object_names(self):
        return self.scene.get_known_object_names()

    def get_attached_object_names(self):
        return self.scene.get_attached_objects().keys()


if __name__ == "__main__":
    rospy.init_node('mico_moveit')

    mm = MicoMoveit()
    print(mm.get_arm_eff_pose())
