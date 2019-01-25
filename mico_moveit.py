import moveit_commander
import rospy

from threading import Lock

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
    ARM_JOINT_NAMES = ['m1n6s200_joint_1',
                       'm1n6s200_joint_2',
                       'm1n6s200_joint_3',
                       'm1n6s200_joint_4',
                       'm1n6s200_joint_5',
                       'm1n6s200_joint_6', ]
    GRIPPER = "gripper"

    OPEN_POSITION = [0] * 2
    CLOSED_POSITION = [1.1] * 2
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

    rospy.loginfo('connected to controllers.')

    def get_arm_eff_pose(self):
        """ Return [[x, y, x], [x, y, z, w]]"""
        pose_stamped = self.arm_commander_group.get_current_pose()
        pose = pose_stamped.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return [position, orientation]

    def get_arm_ik(self, pose_2d, timeout=3):
        """
        Compute arm IK.
        :param pose_2d: 2d list, [[x, y, z], [x, y, z, w]]
        :param timeout: timeout in seconds
        :return: a list of joint values if success; None if no ik
        """
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
        service_request.avoid_collisions = True

        try:
            resp = self.arm_ik_svr(ik_request=service_request)
            if resp.error_code.val == -31:
                print("No ik exixts!")
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


if __name__ == "__main__":
    rospy.init_node('mico_moveit')

    mm = MicoMoveit()
    print(mm.get_arm_eff_pose())