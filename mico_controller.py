import pybullet as p
from collections import namedtuple
from mico_moveit import MicoMoveit
import time
import numpy as np

class MicoController(object):
    ### NOTE 'm1n6s200_joint_1', 'm1n6s200_joint_4', 'm1n6s200_joint_5', 'm1n6s200_joint_6' are circular/continuous joints
    ### NOTE 'm1n6s200_joint_2', 'm1n6s200_joint_3' are not circular/continuous joints
    ### moveit uses a range of [-pi, pi] for circular/continuous joints



    JOINT_TYPES = {
        p.JOINT_REVOLUTE: 'revolute',  # 0
        p.JOINT_PRISMATIC: 'prismatic',  # 1
        p.JOINT_SPHERICAL: 'spherical',  # 2
        p.JOINT_PLANAR: 'planar',  # 3
        p.JOINT_FIXED: 'fixed',  # 4
        p.JOINT_POINT2POINT: 'point2point',  # 5
        p.JOINT_GEAR: 'gear',  # 6
    }

    JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                         'qIndex', 'uIndex', 'flags',
                                         'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                         'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                         'parentFramePos', 'parentFrameOrn', 'parentIndex'])

    JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                           'jointReactionForces', 'appliedJointMotorTorque'])

    LinkState = namedtuple('LinkState', ['linkWorldPosition', 'linkWorldOrientation',
                                         'localInertialFramePosition', 'localInertialFrameOrientation',
                                         'worldLinkFramePosition', 'worldLinkFrameOrientation'])

    # movable joints for each moveit group
    GROUPS = {
        'arm': ['m1n6s200_joint_1', 'm1n6s200_joint_2',
                'm1n6s200_joint_3', 'm1n6s200_joint_4',
                'm1n6s200_joint_5', 'm1n6s200_joint_6', ],
        'gripper': ['m1n6s300_joint_finger_1', 'm1n6s300_joint_finger_2']
    }

    GROUP_INDEX = {
        'arm': [2, 3, 4, 5, 6, 7],
        'gripper': [9, 11]
    }

    ARM_EEF_NAME = "m1n6s300_end_effector"
    ARM_EEF_INDEX = 8

    HOME = [4.80469, 2.92482, 1.002, 4.20319, 1.4458, 1.3233]

    def __init__(self, mico_id):
        self.id = mico_id
        self.num_joints = p.getNumJoints(self.id)
        self.mico_moveit = MicoMoveit()

    ### Control

    def move_arm_eef_pose(self, pose):
        pose_joint_values = self.get_arm_ik(pose)
        self.move_arm_joint_values(pose_joint_values)

    def move_arm_joint_values(self, goal_joint_values):
        start_joint_values = self.get_arm_joint_values()
        start_joint_values = MicoMoveit.convert_range(start_joint_values)
        goal_joint_values = MicoMoveit.convert_range(goal_joint_values)

        plan = self.mico_moveit.plan(start_joint_values, goal_joint_values)
        position_trajectory, velocity_trajectory, time_trajectory_delta = self.mico_moveit.convert_plan(plan)
        # position_trajectory has start_joint_values but not goal_joint_values

        ## NOTE, if you really want to be more precise
        ## 1. add some time for the arm to reach first position in trajecotry
        ## 2. manually add the goal_joint_values to the trajectory
        for i, t in enumerate(time_trajectory_delta):
            p.setJointMotorControlArray(self.id, self.GROUP_INDEX['arm'], p.POSITION_CONTROL, position_trajectory[i], forces=[5000]*len(self.GROUP_INDEX['arm']))
            # time.sleep(t)
            # instead of sleeping t
            time.sleep(0.1)

    def reset_arm_joint_values(self, joint_values):
        joint_values = MicoMoveit.convert_range(joint_values)
        arm_idx = [self.get_joint_names().index(n) for n in self.GROUPS['arm']]
        for i, v in zip(arm_idx, joint_values):
            p.resetJointState(self.id, i, v)

    def reset_arm_eef_pose(self, pose):
        pass

    def move_gripper_joint_values(self, joint_values):
        start_joint_values = self.get_gripper_joint_values()
        goal_joint_values = joint_values
        step = 100
        a = np.linspace(start_joint_values[0], goal_joint_values[0], step)
        b = np.linspace(start_joint_values[1], goal_joint_values[1], step)
        position_trajectory = [[i, j] for (i, j) in zip(a, b)]
        duration = 5
        for i in range(step):
            p.setJointMotorControlArray(self.id, self.GROUP_INDEX['gripper'], p.POSITION_CONTROL, position_trajectory[i], forces=[50]*2)
            time.sleep(float(duration)/float(step))

    def open_gripper(self):
        self.move_gripper_joint_values([0.0, 0.0])

    def close_gripper(self):
        self.move_gripper_joint_values([2.0, 2.0])

    ### Helper functions
    def get_arm_eef_pose(self):
        """
        :return: pose_2d, [[x, y, z], [x, y, x, w]]
        """
        link_state = self.get_link_state(self.ARM_EEF_INDEX)
        position = list(link_state.linkWorldPosition)
        orn = list(link_state.linkWorldOrientation)
        return [position, orn]

    def get_arm_ik(self, pose_2d, timeout=3, avoid_collisions=True):
        ## pybullet ik seems problematic and I do not want to deal with it
        # names = self.get_movable_joint_names()
        # values = p.calculateInverseKinematics(self.id, self.ARM_EEF_INDEX, pose[0], pose[1])
        # d = {n:v for (n, v) in zip(names, values)}
        # return [d[n] for n in self.GROUPS['arm']]
        return MicoMoveit.convert_range(self.mico_moveit.get_arm_ik(pose_2d, timeout, avoid_collisions))

    def get_joint_state(self, joint_index):
        return self.JointState(*p.getJointState(self.id, joint_index))

    def get_arm_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['arm']]

    def get_gripper_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['gripper']]

    def get_joint_names(self, group=None):
        if not group:
            return [p.getJointInfo(self.id, i)[1] for i in range(self.num_joints)]
        else:
            return self.GROUPS[group]

    def get_link_names(self):
        return [p.getJointInfo(self.id, i)[12] for i in range(self.num_joints)]

    def get_link_state(self, link):
        return self.LinkState(*p.getLinkState(self.id, link))

    def get_joint_info(self, joint):
        """ Get joint info by index or name """
        if type(joint) is int:
            return self.JointInfo(*p.getJointInfo(self.id, joint))
        elif type(joint) is str:
            return self.JointInfo(*p.getJointInfo(self.id, self.get_joint_index(joint)))

    def get_movable_joint_names(self):
        return [self.get_joint_info(i).jointName for i in range(self.num_joints) if self.get_joint_info(i).jointType!=p.JOINT_FIXED]

    def get_joint_index(self, name):
        return self.get_joint_names().index(name)



