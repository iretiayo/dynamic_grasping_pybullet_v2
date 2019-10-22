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
import rospy
import moveit_commander as mc
from moveit_msgs.srv import GetPositionIK, GetPositionFK


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

    EEF_LINK_INDEX = 0
    OPEN_POSITION = [0.0, 0.0]
    CLOSED_POSITION = [1.1, 1.1]
    LINK6_COM = [-0.002216, -0.000001, -0.058489]
    LIFT_VALUE = 0.2
    HOME = [4.80469, 2.92482, 1.002, 4.20319, 1.4458, 1.3233]

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

    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.robot_id, self.GROUP_INDEX['arm'], joint_values)

    def control_arm_joints(self, joint_values):
        pu.control_joints(self.robot_id, self.GROUP_INDEX['arm'], joint_values)

    def get_pose(self):
        "the pose is for the link6 center of mass"
        return [list(p.getBasePositionAndOrientation(self.robot_id)[0]), list(p.getBasePositionAndOrientation(self.robot_id)[1])]

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


