import moveit_commander
import rospy
from math import pi
from threading import Lock
import numpy as np

from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import rospy


import rospy
import moveit_commander as mc
import moveit_python
from moveit_msgs.msg import MoveItErrorCodes, DisplayTrajectory,PositionIKRequest
from moveit_msgs.srv import GetPositionIK, GetPositionFK

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
import tf2_ros
import tf_conversions
import tf2_kdl


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
        self.arm_fk_svr = rospy.ServiceProxy('compyte_fk', GetPositionFK)

        self.arm_commander_group = mc.MoveGroupCommander('arm')
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
        ## get moveit_start_state
        start_joint_state = JointState()
        start_joint_state.header = Header()
        start_joint_state.header.stamp = rospy.Time.now() # What is the use of time?
        start_joint_state.name = self.ARM_JOINT_NAMES
        start_joint_state.position = start_joint_values
        start_robot_state = RobotState()
        start_robot_state.joint_state = start_joint_state
        self.arm_commander_group.set_start_state(start_robot_state)
        self.arm_commander_group.set_joint_value_target(goal_joint_values)
        plan = self.arm_commander_group.plan()
        return plan

    @staticmethod
    def extract_plan(plan):
        points = plan.joint_trajectory.points
        header = plan.joint_trajectory.header
        joint_names = plan.joint_trajectory.joint_names
        position_trajecotry = []
        velocity_trajectory = []
        time_trajectory = []
        for p in points:
            # position_trajecotry.append(MicoMoveit.convert_range(p.positions))
            position_trajecotry.append(list(p.positions))
            velocity_trajectory.append(list(p.velocities))
            time_trajectory.append(p.time_from_start.to_sec())
        return np.array(position_trajecotry), np.array(velocity_trajectory), np.array(time_trajectory)

    @staticmethod
    def convert_range(joint_values):
        """ Convert continuous joint to range [-pi, pi] """
        circular_idx = [0, 3, 4, 5]
        new_joint_values = []
        for i, v in enumerate(joint_values):
            if v > pi and i in circular_idx:
                new_joint_values.append(v-2*pi)
            elif v < -pi and i in circular_idx:
                new_joint_values.append(v+2*pi)
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

    def get_arm_ik(self, pose_2d, timeout, avoid_collisions, gripper_joint_values):
        """
        Compute arm IK.
        :param pose_2d: 2d list, [[x, y, z], [x, y, z, w]]
        :param timeout: timeout in seconds
        :param avoid_collisions: whether to avoid collisions when computing ik
        :return: a list of joint values if success; None if no ik
        """
        # when there is collision, we need timeout to control the time to search
        rospy.wait_for_service('compute_ik')

        position = pose_2d[0]
        orientation = pose_2d[1]

        gripper_pose_stamped = PoseStamped()
        gripper_pose_stamped.header.frame_id = self.BASE_LINK
        gripper_pose_stamped.header.stamp = rospy.Time.now()
        gripper_pose_stamped.pose = Pose(Point(*position), Quaternion(*orientation))

        service_request = PositionIKRequest()
        service_request.group_name = self.ARM
        service_request.ik_link_name = self.TIP_LINK
        service_request.pose_stamped = gripper_pose_stamped
        service_request.timeout.secs = timeout
        service_request.avoid_collisions = avoid_collisions

        ## set the gripper joint values when computing arm ik
        from moveit_msgs.msg import RobotState
        from sensor_msgs.msg import JointState

        robot_state = RobotState()
        joint_state = JointState()
        joint_state.position = [0] * 6 + gripper_joint_values
        joint_state.name = MicoMoveit.JOINT_NAMES
        joint_state.header.frame_id = "/world"
        robot_state.joint_state = joint_state
        service_request.robot_state = robot_state

        try:
            resp = self.arm_ik_svr(ik_request=service_request)
            if resp.error_code.val == -31:
                print("No ik exists!")
                return None
            elif resp.error_code.val == 1:
                return self.parse_joint_state_arm(resp.solution.joint_state)
            else:
                print("Other errors!")
                return None
        except rospy.ServiceException, e:
            print("Service call failed: %s" % e)

    def parse_joint_state_arm(self, joint_state):
        d = {n: v for (n, v) in zip(joint_state.name, joint_state.position)}
        return [d[n] for n in self.ARM_JOINT_NAMES]


    ### scene and collision
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

    def get_known_object_names(self):
        return self.scene.get_known_object_names()

    def get_attached_object_names(self):
        return self.scene.get_attached_objects().keys()

if __name__ == "__main__":
    rospy.init_node('mico_moveit')

    mm = MicoMoveit()
    print(mm.get_arm_eff_pose())