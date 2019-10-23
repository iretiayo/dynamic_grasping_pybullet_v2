import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import trimesh
import argparse
import grasp_utils as gu
import pybullet_utils as pu
from collections import OrderedDict
import csv
import tqdm
import tf_conversions
import moveit_commander as mc
from moveit_msgs.srv import GetPositionIK, GetPositionFK

import rospy
from moveit_msgs.msg import DisplayTrajectory, PositionIKRequest, RobotState
from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK, GetPositionFK

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header


class MicoController:
    GROUPS = {
        'arm': ['m1n6s200_joint_1',
                'm1n6s200_joint_2',
                'm1n6s200_joint_3',
                'm1n6s200_joint_4',
                'm1n6s200_joint_5',
                'm1n6s200_joint_6'],
        'gripper': ["m1n6s200_joint_finger_1", "m1n6s200_joint_finger_2"]
    }

    GROUP_INDEX = {
        'arm': [2, 3, 4, 5, 6, 7],
        'gripper': [9, 11]
    }

    INDEX_NAME_MAP = {
        0: 'connect_root_and_world',
        1: 'm1n6s200_joint_base',
        2: 'm1n6s200_joint_1',
        3: 'm1n6s200_joint_2',
        4: 'm1n6s200_joint_3',
        5: 'm1n6s200_joint_4',
        6: 'm1n6s200_joint_5',
        7: 'm1n6s200_joint_6',
        8: 'm1n6s200_joint_end_effector',
        9: 'm1n6s200_joint_finger_1',
        10: 'm1n6s200_joint_finger_tip_1',
        11: 'm1n6s200_joint_finger_2',
        12: 'm1n6s200_joint_finger_tip_2',
    }

    EEF_LINK_INDEX = 8
    OPEN_POSITION = [0.0, 0.0]
    CLOSED_POSITION = [1.1, 1.1]
    LINK6_COM = [-0.002216, -0.000001, -0.058489]
    LIFT_VALUE = 0.2
    HOME = [4.80469, 2.92482, 1.002, 4.20319, 1.4458, 1.3233]
    EEF_LINK = "m1n6s200_end_effector"
    BASE_LINK = "root"

    def __init__(self, robot_id):
        self.robot_id = robot_id

        self.arm_ik_svr = rospy.ServiceProxy('compute_ik', GetPositionIK)
        self.arm_fk_svr = rospy.ServiceProxy('compute_fk', GetPositionFK)

        self.arm_commander_group = mc.MoveGroupCommander('arm')
        self.robot = mc.RobotCommander()
        self.scene = mc.PlanningSceneInterface()
        rospy.sleep(2)

        self.eef_link = self.arm_commander_group.get_end_effector_link()
        self.kf = None
        self.freeze_time = 0
        self.current_plan = None

        # the service names have to be this
        self.arm_ik_svr = rospy.ServiceProxy('compute_ik', GetPositionIK)
        self.arm_fk_svr = rospy.ServiceProxy('compute_fk', GetPositionFK)

    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.robot_id, self.GROUP_INDEX['arm'], joint_values)
        pu.control_joints(self.robot_id, self.GROUP_INDEX['arm'], joint_values)

    def control_arm_joints(self, joint_values):
        pu.control_joints(self.robot_id, self.GROUP_INDEX['arm'], joint_values)

    def plan(self, start_joint_values, goal_joint_values, maximum_planning_time=0.5):
        """ No matter what start and goal are, the returned plan start and goal will
            make circular joints within [-pi, pi] """
        # setup moveit_start_state
        start_robot_state = self.robot.get_current_state()
        start_robot_state.joint_state.name = self.GROUPS['arm']
        start_robot_state.joint_state.position = start_joint_values

        self.arm_commander_group.set_start_state(start_robot_state)
        self.arm_commander_group.set_joint_value_target(goal_joint_values)
        self.arm_commander_group.set_planning_time(maximum_planning_time)
        # takes around 0.11 second
        plan = self.arm_commander_group.plan()
        return plan

    def compute_next_action(self, object_pose, ):
        pass

    def step(self):
        """ step the robot for 1/240 second """
        # calculate the latest conf and control array
        pass

    def get_arm_joint_values(self):
        return pu.get_joint_positions(self.robot_id, self.GROUP_INDEX['arm'])

    def get_gripper_joint_values(self):
        return pu.get_joint_positions(self.robot_id, self.GROUP_INDEX['gripper'])

    def get_eef_pose(self):
        return pu.get_link_pose(self.robot_id, self.EEF_LINK_INDEX)

    def get_arm_ik(self, pose_2d, timeout=0.01, avoid_collisions=True):
        gripper_joint_values = self.get_gripper_joint_values()
        arm_joint_values = self.get_arm_joint_values()
        j = self.get_arm_ik_ros(pose_2d, timeout, avoid_collisions, arm_joint_values, gripper_joint_values)
        if j is None:
            # print("No ik exists!")
            return None
        else:
            # return MicoMoveit.convert_range(j)
            return j

    def get_arm_fk(self, arm_joint_values):
        pose = self.get_arm_fk_ros(arm_joint_values)
        return gu.pose_2_list(pose) if pose is not None else None

    def get_arm_ik_ros(self, pose_2d, timeout, avoid_collisions, arm_joint_values, gripper_joint_values):
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
        service_request.group_name = "arm"
        service_request.ik_link_name = self.EEF_LINK
        service_request.pose_stamped = gripper_pose_stamped
        service_request.timeout.secs = timeout
        service_request.avoid_collisions = avoid_collisions

        seed_robot_state = self.robot.get_current_state()
        seed_robot_state.joint_state.name = self.GROUPS['arm'] + self.GROUPS['gripper']
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

    def get_arm_fk_ros(self, arm_joint_values):
        """ return a ros pose """
        rospy.wait_for_service('compute_fk')

        header = Header(frame_id="world")
        fk_link_names = [self.EEF_LINK]
        robot_state = RobotState()
        robot_state.joint_state.name = self.GROUPS['arm']
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
        return [d[n] for n in self.GROUPS['arm']]

    def close_gripper(self):
        num_steps = 240
        waypoints = np.linspace(self.OPEN_POSITION, self.CLOSED_POSITION, num_steps)
        for wp in waypoints:
            pu.control_joints(self.robot_id, self.GROUP_INDEX['gripper'], wp)
            p.stepSimulation()
        pu.step()

