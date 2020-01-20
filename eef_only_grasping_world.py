import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import grasp_utils as gu
import pybullet_utils as pu
from math import pi, cos, sin, sqrt, atan, radians
import copy
from dynamic_grasping_world import Conveyor
import misc_utils as mu
from random import uniform
import tf_conversions


class EEFController:
    """ The controller for EEF only """

    LIFT_VALUE = 0.2

    def __init__(self, robot_config_name, gripper_initial_pose):
        self.load_robot_configs(robot_config_name)
        self.gripper_initial_pose = gripper_initial_pose
        self.id = p.loadURDF(self.gripper_urdf, self.gripper_initial_pose[0], self.gripper_initial_pose[1],
                             flags=p.URDF_USE_SELF_COLLISION)
        self.cid = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                      childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0], childFramePosition=self.gripper_initial_pose[0],
                                      childFrameOrientation=self.gripper_initial_pose[1])
        joint_infos = [p.getJointInfo(self.id, joint_index) for joint_index in range(p.getNumJoints(self.id))]
        self.JOINT_INDICES_DICT = {entry[1]: entry[0] for entry in joint_infos}
        self.GRIPPER_INDICES = [self.JOINT_INDICES_DICT[name] for name in self.GRIPPER_JOINT_NAMES]

        # step plans
        self.arm_discretized_plan = None
        self.gripper_discretized_plan = None
        self.arm_wp_target_index = 0
        self.gripper_wp_target_index = 0

    def load_robot_configs(self, robot_config_name):
        self.robot_config_name = robot_config_name
        self.robot_configs = gu.robot_configs[self.robot_config_name]
        self.gripper_urdf = self.robot_configs.gripper_urdf
        self.EEF_LINK_INDEX = self.robot_configs.EEF_LINK_INDEX
        self.GRIPPER_JOINT_NAMES = self.robot_configs.GRIPPER_JOINT_NAMES
        self.OPEN_POSITION = self.robot_configs.OPEN_POSITION
        self.CLOSED_POSITION = self.robot_configs.CLOSED_POSITION

        self.GRASPIT_LINK_TO_MOVEIT_LINK = self.robot_configs.GRASPIT_LINK_TO_MOVEIT_LINK
        self.GRASPIT_LINK_TO_PYBULLET_LINK = self.robot_configs.GRASPIT_LINK_TO_PYBULLET_LINK
        self.PYBULLET_LINK_TO_COM = self.robot_configs.PYBULLET_LINK_TO_COM

        self.GRASPIT_LINK_TO_PYBULLET_LINK_COM = tf_conversions.toTf(
            tf_conversions.fromTf(self.GRASPIT_LINK_TO_PYBULLET_LINK) * tf_conversions.fromTf(
                self.PYBULLET_LINK_TO_COM))
        self.PYBULLET_LINK_COM_TO_GRASPIT_LINK = tf_conversions.toTf(
            tf_conversions.fromTf(self.GRASPIT_LINK_TO_PYBULLET_LINK_COM).Inverse())

    def reset(self):
        # step plans
        self.arm_discretized_plan = None
        self.gripper_discretized_plan = None
        self.arm_wp_target_index = 0
        self.gripper_wp_target_index = 0
        self.reset_to(self.gripper_initial_pose)
        self.open_gripper()

    def reset_to(self, pose):
        """ the pose is for the link6 center of mass """
        p.resetBasePositionAndOrientation(self.id, pose[0], pose[1])
        self.move_to(pose)

    def move_to(self, pose):
        """ the pose is for the link6 center of mass """
        num_steps = 240
        current_pose = self.get_pose()
        positions = np.linspace(current_pose[0], pose[0], num_steps)
        angles = np.linspace(p.getEulerFromQuaternion(current_pose[1]), p.getEulerFromQuaternion(pose[1]), num_steps)
        quaternions = np.array([p.getQuaternionFromEuler(angle) for angle in angles])
        for pos, ori in zip(positions, quaternions):
            self.control_pose([pos, ori])
            p.stepSimulation()
        pu.step()

    def control_pose(self, pose):
        """ the pose is for the link6 center of mass """
        p.changeConstraint(self.cid, jointChildPivot=pose[0], jointChildFrameOrientation=pose[1])

    def close_gripper(self):
        num_steps = 240
        waypoints = np.linspace(self.OPEN_POSITION, self.CLOSED_POSITION, num_steps)
        for wp in waypoints:
            pu.control_joints(self.id, self.GRIPPER_INDICES, wp)
            p.stepSimulation()
        pu.step()

    def get_current_grasp_pose(self):
        pybullet_base_com_pose_2d = pu.get_link_pose(self.id, -1)
        graspit_pose_2d = gu.change_end_effector_link_pose_2d(pybullet_base_com_pose_2d, self.PYBULLET_LINK_COM_TO_GRASPIT_LINK)
        return graspit_pose_2d

    def execute_grasp(self, grasp, back_off):
        """ High level grasp interface using grasp 2d in world frame (link6_reference_frame)"""
        com_pose_2d = gu.change_end_effector_link_pose_2d(grasp, self.GRASPIT_LINK_TO_PYBULLET_LINK_COM)
        pregrasp_com_pose_2d = gu.back_off_pose_2d(com_pose_2d, back_off)

        self.reset_to(pregrasp_com_pose_2d)
        actual_pregrasp = self.get_current_grasp_pose()

        self.move_to(com_pose_2d)
        actual_grasp = self.get_current_grasp_pose()

        self.close_gripper()
        self.lift()
        # robust test
        self.lift(0.2)
        self.lift(-0.2)
        pu.step(2)
        return actual_pregrasp, actual_grasp

    def execute_grasp_link6_com(self, grasp):
        """ High level grasp interface using grasp 2d in world frame (link6_com_frame)"""
        self.reset_to(grasp)
        self.close_gripper()
        self.lift()
        pu.step(2)

    def execute_grasp_link6_com_with_pre_grasp(self, grasp, pre_grasp):
        """ High level grasp interface using grasp 2d in world frame (link6_com_frame)"""
        self.reset_to(pre_grasp)
        self.move_to(grasp)
        self.close_gripper()
        self.lift()
        self.lift(0.2)
        self.lift(-0.2)
        pu.step(2)

    def open_gripper(self):
        pu.set_joint_positions(self.id, self.GRIPPER_INDICES, self.OPEN_POSITION)
        pu.control_joints(self.id, self.GRIPPER_INDICES, self.OPEN_POSITION)
        pu.step()

    def lift(self, z=LIFT_VALUE):
        target_pose = self.get_pose()
        target_pose[0][2] += z
        self.move_to(target_pose)

    def get_pose(self):
        """ the pose is for the link6 center of mass """
        return [list(p.getBasePositionAndOrientation(self.id)[0]),
                list(p.getBasePositionAndOrientation(self.id)[1])]

    def initialize_gripper_plan(self, start_joint_values, goal_joint_values):
        num_steps = 240
        self.gripper_discretized_plan = np.linspace(start_joint_values, goal_joint_values, num_steps)
        self.gripper_wp_target_index = 1

    def initialize_hand_plan(self, end_eef_pose, start_eef_pose=None):
        num_steps = 240
        start_eef_pose = self.get_pose() if start_eef_pose is None else start_eef_pose
        positions = np.linspace(start_eef_pose[0], end_eef_pose[0], num_steps)
        angles = np.linspace(p.getEulerFromQuaternion(start_eef_pose[1]), p.getEulerFromQuaternion(end_eef_pose[1]), num_steps)
        quaternions = np.array([p.getQuaternionFromEuler(angle) for angle in angles])
        self.arm_discretized_plan = [[list(pos), list(quat)] for pos, quat in zip(positions, quaternions)]
        self.arm_wp_target_index = 1

    def step(self):
        if self.arm_discretized_plan is not None and self.arm_wp_target_index != len(self.arm_discretized_plan):
            self.control_pose(self.arm_discretized_plan[self.arm_wp_target_index])
            self.arm_wp_target_index += 1
        if self.gripper_discretized_plan is not None and self.gripper_wp_target_index != len(self.gripper_discretized_plan):
            pu.control_joints(self.id, self.GRIPPER_INDICES, self.gripper_discretized_plan[self.gripper_wp_target_index])
            self.gripper_wp_target_index += 1


class EEFOnlyStaticWorld:
    """ EEF only world with static target """

    def __init__(self, target_initial_pose, gripper_initial_pose, robot_config_name, target_urdf, apply_noise):
        self.target_initial_pose = target_initial_pose
        self.gripper_initial_pose = gripper_initial_pose
        self.robot_config_name = robot_config_name
        self.target_urdf = target_urdf
        self.apply_noise = apply_noise
        self.x_noise = 0.01
        self.y_noise = 0.01
        self.rotation_noise = 20

        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])
        self.controller = EEFController(self.robot_config_name, self.gripper_initial_pose)

    def reset(self):
        p.resetBasePositionAndOrientation(self.target, self.target_initial_pose[0], self.target_initial_pose[1])
        self.controller.reset_to(self.gripper_initial_pose)
        self.controller.open_gripper()
        if self.apply_noise:
            sampled_x_noise = np.random.uniform(low=-self.x_noise, high=self.x_noise)
            sampled_y_noise = np.random.uniform(low=-self.y_noise, high=self.y_noise)
            sampled_rotation_noise = np.random.uniform(low=-self.rotation_noise, high=self.rotation_noise)
            target_noise_pose = copy.deepcopy(pu.get_body_pose(self.target))
            target_noise_pose[0][0] += sampled_x_noise
            target_noise_pose[0][1] += sampled_y_noise
            target_rpy = pu.get_euler_from_quaternion(target_noise_pose[1])
            target_rpy[2] += radians(sampled_rotation_noise)
            target_noise_pose[1] = pu.get_quaternion_from_euler(target_rpy)
            p.resetBasePositionAndOrientation(self.target, target_noise_pose[0], target_noise_pose[1])
            pu.step()


class EEFOnlyDynamicWorld:
    """ EEF only world with dynamic target, for motion aware grasp planning """

    def __init__(self, target_initial_pose,
                 conveyor_initial_pose,
                 gripper_initial_pose,
                 gripper_urdf,
                 target_urdf,
                 conveyor_urdf,
                 min_speed,
                 max_speed):
        self.target_initial_pose = target_initial_pose
        self.conveyor_initial_pose = conveyor_initial_pose
        self.gripper_initial_pose = gripper_initial_pose
        self.gripper_urdf = gripper_urdf
        self.target_urdf = target_urdf
        self.conveyor_urdf = conveyor_urdf
        self.min_speed = min_speed
        self.max_speed = max_speed

        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])

        self.controller = EEFController(self.gripper_urdf, self.gripper_initial_pose)
        self.conveyor = Conveyor(self.conveyor_initial_pose, self.conveyor_urdf)

    def reset(self, reset_dict=None):
        pu.remove_all_markers()
        self.conveyor.clear_motion()    # this is not necessary
        if reset_dict is not None:
            angle, speed, distance = reset_dict['angle'], reset_dict['speed'], reset_dict['distance']
        else:
            angle = uniform(0, 360)
            speed = uniform(self.min_speed, self.max_speed)
            distance = 0.5

        self.conveyor.set_pose(self.conveyor_initial_pose)
        p.resetBasePositionAndOrientation(self.target, self.target_initial_pose[0], self.target_initial_pose[1])
        self.controller.reset()
        pu.step(2)

        # TODO a more precise way of calculating the movement
        # conveyor_pose = pu.get_body_pose(self.conveyor.id)
        # object_pose = pu.get_body_pose(self.target)
        # conveyor_T_object = mu.calculate_transform(conveyor_pose, object_pose)
        # object_target_pose_in_object, object_target_pose_in_world = mu.calculate_target_pose(object_pose, angle, distance)
        # conveyor_target_pose_in_conveyor =

        conveyor_start_pose, conveyor_target_pose = self.conveyor.initialize_linear_motion_v2(angle, speed, distance)
        pu.draw_line(conveyor_start_pose[0], conveyor_target_pose[0])
        return angle, speed

    def step(self):
        self.conveyor.step()
        self.controller.step()
        p.stepSimulation()
        # time.sleep(1.0/240.0)

    def dynamic_grasp(self, grasp_link6_com_in_object, pre_grasp_link6_com_in_object):

        object_pose = p.getBasePositionAndOrientation(self.target)
        success_height_threshold = object_pose[0][2] + self.controller.LIFT_VALUE - 0.05
        grasp_link6_com_in_object = pu.split_7d(grasp_link6_com_in_object)
        grasp_link6_com_in_world = gu.convert_grasp_in_object_to_world(object_pose, grasp_link6_com_in_object)
        pre_grasp_link6_com_in_object = pu.split_7d(pre_grasp_link6_com_in_object)
        pre_grasp_link6_com_in_world = gu.convert_grasp_in_object_to_world(object_pose, pre_grasp_link6_com_in_object)

        self.controller.reset_to(pre_grasp_link6_com_in_world)
        self.controller.initialize_hand_plan(grasp_link6_com_in_world, pre_grasp_link6_com_in_world, )
        self.controller.initialize_gripper_plan(self.OPEN_POSITION, self.CLOSED_POSITION)

        done = False
        while not done:
            self.step()
            done = self.check_done()
        self.controller.lift()
        self.controller.lift(0.2)
        self.controller.lift(-0.2)
        success = p.getBasePositionAndOrientation(self.target)[0][2] > success_height_threshold
        return success

    def check_done(self):
        done = False
        if self.controller.arm_wp_target_index == len(self.controller.arm_discretized_plan) \
                and self.controller.gripper_wp_target_index == len(self.controller.gripper_discretized_plan):
            done = True
        if self.conveyor.wp_target_index == len(self.conveyor.discretized_trajectory):
            # conveyor trajectory finishes
            done = True
        if pu.get_body_pose(self.target)[0][2] < self.target_initial_pose[0][2] - 0.03:
            # target knocked down
            done = True
        return done