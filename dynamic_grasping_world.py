import os
import numpy as np
import pyquaternion as pyqt
import pybullet as p
import pybullet_data
import time
import grasp_utils as gu
import pybullet_utils as pu
from mico_controller import MicoController
import rospy
import threading
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from math import pi, cos, sin, sqrt, atan, radians, degrees
from kalman_filter_3d import KalmanFilter, create_kalman_filter
import random
import tf_conversions as tfc
import moveit_commander as mc
from ur5_robotiq_pybullet import load_ur_robotiq_robot, UR5RobotiqPybulletController
from shapely.geometry import Polygon, Point
import misc_utils as mu
import trimesh
from itertools import combinations
from train_motion_aware import MotionQualityEvaluationNet
import torch
from torch.nn.functional import softmax


class DynamicGraspingWorld:
    def __init__(self,
                 target_name,
                 obstacle_names,
                 mesh_dir,
                 robot_config_name,
                 target_initial_pose,
                 robot_initial_pose,
                 robot_initial_state,
                 conveyor_initial_pose,
                 robot_urdf,
                 conveyor_urdf,
                 conveyor_speed,
                 target_urdf,
                 target_mesh_file_path,
                 target_extents,
                 grasp_database_path,
                 reachability_data_dir,
                 realtime,
                 max_check,
                 disable_reachability,
                 back_off,
                 pose_freq,
                 use_seed_trajectory,
                 use_previous_jv,
                 use_kf,
                 use_gt,
                 grasp_threshold,
                 lazy_threshold,
                 large_prediction_threshold,
                 small_prediction_threshold,
                 close_delay,
                 distance_travelled_threshold,
                 distance_low,
                 distance_high,
                 circular_distance_low,
                 circular_distance_high,
                 use_box,
                 use_baseline_method,
                 approach_prediction,
                 approach_prediction_duration,
                 fix_motion_planning_time,
                 fix_grasp_ranking_time,
                 load_obstacles,
                 obstacle_distance_low,
                 obstacle_distance_high,
                 distance_between_region,
                 use_motion_aware,
                 motion_aware_model_path):
        self.target_name = target_name
        self.obstacle_names = obstacle_names
        self.mesh_dir = mesh_dir
        self.robot_config_name = robot_config_name
        self.target_initial_pose = target_initial_pose
        self.robot_initial_pose = robot_initial_pose
        self.initial_distance = np.linalg.norm(
            np.array(target_initial_pose[0][:2]) - np.array(robot_initial_pose[0][:2]))
        self.robot_initial_state = robot_initial_state
        self.conveyor_initial_pose = conveyor_initial_pose
        self.robot_urdf = robot_urdf
        self.conveyor_urdf = conveyor_urdf
        self.conveyor_speed = conveyor_speed
        self.target_urdf = target_urdf
        self.target_mesh_file_path = target_mesh_file_path
        self.target_extents = target_extents
        self.realtime = realtime
        self.max_check = max_check
        self.back_off = back_off
        self.disable_reachability = disable_reachability
        self.world_steps = 0
        self.pose_freq = pose_freq
        self.pose_duration = 1.0 / self.pose_freq
        self.pose_steps = int(self.pose_duration * 240)
        self.use_seed_trajectory = use_seed_trajectory
        self.use_previous_jv = use_previous_jv
        self.use_kf = use_kf
        self.use_gt = use_gt
        self.motion_predictor_kf = MotionPredictorKF(self.pose_duration)
        self.distance_between_region = distance_between_region
        self.use_motion_aware = use_motion_aware
        self.motion_aware_model_path = motion_aware_model_path

        self.distance_low = distance_low  # mico 0.15  ur5_robotiq: 0.3
        self.distance_high = distance_high  # mico 0.4  ur5_robotiq: 0.7
        self.circular_distance_low = circular_distance_low
        self.circular_distance_high = circular_distance_high

        self.grasp_database_path = grasp_database_path
        actual_grasps, graspit_grasps = gu.load_grasp_database_new(grasp_database_path, self.target_name)
        use_actual = False
        self.graspit_grasps = actual_grasps if use_actual else graspit_grasps

        self.robot_configs = gu.robot_configs[self.robot_config_name]
        self.graspit_pregrasps = [
            pu.merge_pose_2d(gu.back_off_pose_2d(pu.split_7d(g), back_off, self.robot_configs.graspit_approach_dir)) for
            g in self.graspit_grasps]
        self.grasps_eef = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose_2d(pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_MOVEIT_LINK)) for g
            in self.graspit_grasps]
        self.grasps_link6_ref = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose_2d(pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_PYBULLET_LINK)) for g
            in self.graspit_grasps]
        self.pre_grasps_eef = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose_2d(pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_MOVEIT_LINK)) for g
            in self.graspit_pregrasps]
        self.pre_grasps_link6_ref = [pu.merge_pose_2d(
            gu.change_end_effector_link_pose_2d(pu.split_7d(g), self.robot_configs.GRASPIT_LINK_TO_PYBULLET_LINK)) for g
            in self.graspit_pregrasps]

        self.reachability_data_dir = reachability_data_dir
        self.sdf_reachability_space, self.mins, self.step_size, self.dims = gu.get_reachability_space(
            self.reachability_data_dir)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])
        if 'mico' in self.robot_config_name:
            self.robot = MicoController(self.robot_initial_pose, self.robot_initial_state, self.robot_urdf)
        if 'robotiq' in self.robot_config_name:
            self.robot_id = load_ur_robotiq_robot(self.robot_initial_pose)
            self.robot = UR5RobotiqPybulletController(self.robot_id)
            p.changeDynamics(self.target, -1, mass=1)
            for joint in range(p.getNumJoints(self.robot.id)):
                p.changeDynamics(self.robot.id, joint, mass=1)
        self.conveyor = Conveyor(self.conveyor_initial_pose, self.conveyor_urdf)
        self.reset('initial')  # the reset is needed to simulate the initial config

        self.scene = mc.PlanningSceneInterface()
        self.grasp_threshold = grasp_threshold
        self.close_delay = close_delay
        self.lazy_threshold = lazy_threshold
        self.large_prediction_threshold = large_prediction_threshold
        self.small_prediction_threshold = small_prediction_threshold
        self.distance_travelled_threshold = distance_travelled_threshold
        self.use_box = use_box
        self.use_baseline_method = use_baseline_method
        self.approach_prediction = approach_prediction
        self.approach_prediction_duration = approach_prediction_duration
        self.fix_motion_planning_time = fix_motion_planning_time
        self.fix_grasp_ranking_time = fix_grasp_ranking_time

        # obstacles
        self.load_obstacles = load_obstacles
        self.obstacle_distance_low = obstacle_distance_low
        self.obstacle_distance_high = obstacle_distance_high

        self.obstacles = []
        if self.load_obstacles:
            self.obstacle_urdfs = []
            self.obstacle_zs = []
            self.obstacle_extentss = []
            for obstacle_name in self.obstacle_names:
                mesh_filepath = os.path.join(self.mesh_dir, '{}'.format(obstacle_name), '{}.obj'.format(obstacle_name))
                self.obstacle_urdfs.append(mu.create_object_urdf(mesh_filepath, obstacle_name,
                                                                 urdf_target_object_filepath='assets/{}_obstacle.urdf'.format(
                                                                     obstacle_name)))
                obstacle_mesh = trimesh.load_mesh(mesh_filepath)
                self.obstacle_extentss.append(obstacle_mesh.bounding_box.extents.tolist())
                self.obstacle_zs.append(-obstacle_mesh.bounds.min(0)[2])

        if self.use_motion_aware:
            self.motion_aware_network = MotionQualityEvaluationNet()
            epoch_dir = os.listdir(os.path.join(self.motion_aware_model_path, self.target_name))[0]
            self.motion_aware_network.load_state_dict(torch.load(
                os.path.join(self.motion_aware_model_path, self.target_name, epoch_dir,
                             'motion_ware_net.pt')))

    def reset(self, mode, reset_dict=None):
        """
        mode:
            initial: reset the target to the fixed initial pose, not moving
            static_random: reset the target to a random pose, not moving
            dynamic_linear: initialize the conveyor with a linear motion
            dynamic_circular: initialize the conveyor with a circular motion
            hand_over: TODO
        """
        self.world_steps = 0
        if mode == 'initial':
            pu.remove_all_markers()
            target_pose, distance = self.target_initial_pose, self.initial_distance
            conveyor_pose = [[target_pose[0][0], target_pose[0][1], 0.01],
                             [0, 0, 0, 1]] if target_pose is not None else self.conveyor_initial_pose
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.set_pose(conveyor_pose)
            self.robot.reset()
            pu.step(2)
            return target_pose, distance

        elif mode == 'static_random':
            pu.remove_all_markers()
            target_pose, distance = self.sample_target_location()
            conveyor_pose = [[target_pose[0][0], target_pose[0][1], 0.01],
                             [0, 0, 0, 1]] if target_pose is not None else self.conveyor_initial_pose
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.set_pose(conveyor_pose)
            self.robot.reset()
            self.scene.add_box("floor", gu.list_2_ps(((0, 0, -0.055), (0, 0, 0, 1))), size=(2, 2, 0.1))
            pu.step(2)
            return target_pose, distance

        elif mode == 'dynamic_linear':
            pu.remove_all_markers()
            if len(self.obstacles) != 0:
                for i in self.obstacles:
                    p.removeBody(i)
            self.motion_predictor_kf.reset_predictor()
            self.conveyor.clear_motion()

            if reset_dict is None:
                distance, theta, length, direction = self.sample_convey_linear_motion()
                target_quaternion = self.sample_target_angle()
            else:
                distance, theta, length, direction = reset_dict['distance'], reset_dict['theta'], reset_dict['length'], \
                                                     reset_dict['direction']
                target_quaternion = reset_dict['target_quaternion']
            self.conveyor.initialize_linear_motion(distance, theta, length, direction, self.conveyor_speed)
            conveyor_pose = self.conveyor.start_pose
            target_pose = [[conveyor_pose[0][0], conveyor_pose[0][1], self.target_initial_pose[0][2]],
                           target_quaternion]
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.set_pose(conveyor_pose)
            if self.load_obstacles:
                if reset_dict['obstacle_poses'] is None:
                    self.obstacles = self.load_obstacles_collision_free(distance, theta, length)
                else:
                    # self.get_obstacles_regions(distance, theta, length, visualize_region=True)
                    self.obstacles = self.load_obstacles_at_poses(reset_dict['obstacle_poses'])
            self.robot.reset()
            # self.scene.add_box("floor", gu.list_2_ps(((0, 0, -0.055), (0, 0, 0, 1))), size=(2, 2, 0.1))
            pu.step(2)

            obstacle_poses = []
            if self.load_obstacles:
                for i, n, e in zip(self.obstacles, self.obstacle_names, self.obstacle_extentss):
                    self.scene.add_box(n, gu.list_2_ps(pu.get_body_pose(i)), size=e)
                    obstacle_poses.append(pu.merge_pose_2d(pu.get_body_pose(i)))

            self.motion_predictor_kf.initialize_predictor(target_pose)
            pu.draw_line(self.conveyor.start_pose[0], self.conveyor.target_pose[0])
            return distance, theta, length, direction, target_quaternion, obstacle_poses

        elif mode == 'dynamic_circular':
            pu.remove_all_markers()
            if len(self.obstacles) != 0:
                for i in self.obstacles:
                    p.removeBody(i)
            self.motion_predictor_kf.reset_predictor()
            self.conveyor.clear_motion()

            if reset_dict is None:
                distance, theta, length, direction = self.sample_convey_circular_motion()
                target_quaternion = self.sample_target_angle()
            else:
                distance, theta, length, direction = reset_dict['distance'], reset_dict['theta'], reset_dict['length'], \
                                                     reset_dict['direction']
                target_quaternion = reset_dict['target_quaternion']
            self.conveyor.initialize_circular_motion(distance, theta, length, direction, self.conveyor_speed)
            conveyor_pose = self.conveyor.start_pose
            target_pose = [[conveyor_pose[0][0], conveyor_pose[0][1], self.target_initial_pose[0][2]],
                           target_quaternion]
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.set_pose(conveyor_pose)
            self.robot.reset()
            # self.scene.add_box("floor", gu.list_2_ps(((0, 0, -0.055), (0, 0, 0, 1))), size=(2, 2, 0.1))
            pu.step(2)

            obstacle_poses = []
            self.motion_predictor_kf.initialize_predictor(target_pose)

            # visualize circular motion, evenly pick points in the trajectory
            num_plot_points = 100
            idx = np.round(np.linspace(0, len(self.conveyor.discretized_trajectory) - 1, num_plot_points)).astype(int)
            for i in range(len(idx) - 1):
                pos1 = self.conveyor.discretized_trajectory[idx[i]][0]
                pos2 = self.conveyor.discretized_trajectory[idx[i+1]][0]
                pu.draw_line(pos1, pos2)

            return distance, theta, length, direction, target_quaternion, obstacle_poses

        elif mode == 'hand_over':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def step(self, freeze_time, arm_motion_plan, gripper_motion_plan):
        for i in range(int(freeze_time * 240)):
            # step the robot
            self.robot.step()
            # step conveyor
            self.conveyor.step()
            # step physics
            p.stepSimulation()
            self.world_steps += 1
            if self.world_steps % self.pose_steps == 0:
                self.motion_predictor_kf.update(pu.get_body_pose(self.target))
            if self.realtime:
                time.sleep(1.0 / 240.0)
        if arm_motion_plan is not None:
            self.robot.update_arm_motion_plan(arm_motion_plan)
        if gripper_motion_plan is not None:
            self.robot.update_gripper_motion_plan(gripper_motion_plan)

    def static_grasp(self):
        target_pose = pu.get_body_pose(self.target)
        predicted_pose = target_pose

        success = False
        grasp_attempted = False  # pre_grasp and grasp is reachable and motion is found
        pre_grasp_reached = False
        grasp_reachaed = False
        comment = " "

        # planning grasp
        grasp_idx, grasp_planning_time, num_ik_called, pre_grasp, pre_grasp_jv, grasp, grasp_jv, grasp_switched = self.plan_grasp(
            predicted_pose, None)
        if grasp_jv is None or pre_grasp_jv is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, "no reachable grasp is found"

        # planning motion
        if self.use_box:
            self.scene.add_box('target', gu.list_2_ps(target_pose), size=self.target_extents)
        else:
            self.scene.add_mesh('target', gu.list_2_ps(target_pose), self.target_mesh_file_path)
        rospy.sleep(2)
        motion_planning_time, plan = self.plan_arm_motion(pre_grasp_jv)
        if plan is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, "no motion found to the planned pre grasp jv"

        # move
        self.robot.execute_arm_plan(plan, self.realtime)
        pre_grasp_reached = self.robot.equal_conf(self.robot.get_arm_joint_values(), pre_grasp_jv, tol=0.01)

        # print('self')
        # print(self.robot.get_arm_joint_values())
        # print('pre_grasp_jv')
        # print(pre_grasp_jv)
        # print('grasp_jv')
        # print(grasp_jv)

        # approach
        # plan = self.robot.plan_arm_joint_values_simple(grasp_jv)
        # self.robot.execute_arm_plan(plan, self.realtime)
        # grasp_reachaed = self.robot.equal_conf(self.robot.get_arm_joint_values(), grasp_jv, tol=0.01)
        plan, fraction = self.robot.plan_straight_line(tfc.toMsg(tfc.fromTf(grasp)), ee_step=0.01,
                                                       avoid_collisions=True)
        if plan is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, "no motion found to the planned grasp jv"
        self.robot.execute_arm_plan(plan, self.realtime)
        # print(fraction)
        grasp_attempted = True

        # close and lift
        self.robot.close_gripper(self.realtime)
        plan, fraction = self.robot.plan_cartesian_control(z=0.07)
        if fraction != 1.0:
            comment = "lift fration {} is not 1.0".format(fraction)
        if plan is not None:
            self.robot.execute_arm_plan(plan, self.realtime)
        success = self.check_success()
        pu.remove_all_markers()
        return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, comment

    def check_success(self):
        if pu.get_body_pose(self.target)[0][2] >= self.target_initial_pose[0][2] + 0.03:
            return True
        else:
            return False

    def predict(self, duration):
        if self.use_kf:
            # TODO verify that when duration is 0
            predicted_target_pose = self.motion_predictor_kf.predict(duration)
            predicted_conveyor_position = list(predicted_target_pose[0])
            predicted_conveyor_position[2] = 0.01
            predicted_conveyor_pose = [predicted_conveyor_position, [0, 0, 0, 1]]
        elif self.use_gt:
            current_target_pose = pu.get_body_pose(self.target)
            predicted_conveyor_pose = self.conveyor.predict(duration)
            predicted_target_position = [predicted_conveyor_pose[0][0], predicted_conveyor_pose[0][1],
                                         current_target_pose[0][2]]
            predicted_target_pose = [predicted_target_position, current_target_pose[1]]
        else:
            # no prediction
            predicted_target_pose = pu.get_body_pose(self.target)
            predicted_conveyor_pose = pu.get_body_pose(self.conveyor.id)
        return predicted_target_pose, predicted_conveyor_pose

    def can_grasp(self, grasp_idx):
        planned_grasp_in_object = pu.split_7d(self.grasps_eef[grasp_idx])
        grasp_pose_tf = gu.convert_grasp_in_object_to_world(pu.get_body_pose(self.target), planned_grasp_in_object)
        current_eef_pose_tf = self.robot.get_eef_pose()

        dist_pos = np.abs(np.array(grasp_pose_tf[0]) - np.array(current_eef_pose_tf[0]))
        dist_q = pyqt.Quaternion.absolute_distance(pyqt.Quaternion(grasp_pose_tf[1]),
                                                   pyqt.Quaternion(current_eef_pose_tf[1]))
        can_grasp = np.linalg.norm(dist_pos) < np.abs(self.back_off * 1.1) and np.abs(dist_q) < np.pi / 180 * 20.
        return can_grasp

    def dynamic_grasp(self):
        """

        :return attempted_grasp_idx: the executed grasp index
        """
        grasp_idx = None
        done = False
        dynamic_grasp_time = 0
        distance = None
        initial_motion_plan_success = False  # not necessarily succeed
        while not done:
            done = self.check_done()
            current_target_pose = pu.get_body_pose(self.target)
            duration = self.calculate_prediction_time(distance)
            predicted_target_pose, predicted_conveyor_pose = self.predict(duration)

            # update the scene. it will not reach the next line if the scene is not updated
            update_start_time = time.time()
            if self.use_box:
                self.scene.add_box('target', gu.list_2_ps(predicted_target_pose), size=self.target_extents)
            else:
                self.scene.add_mesh('target', gu.list_2_ps(predicted_target_pose), self.target_mesh_file_path)
            self.scene.add_box('conveyor', gu.list_2_ps(predicted_conveyor_pose), size=(.1, .1, .02))
            # print('Updating scene takes {} second'.format(time.time() - update_start_time))

            # plan a grasp
            if self.use_baseline_method:
                grasp_idx, grasp_planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv, grasp_switched \
                    = self.plan_grasp_baseline(predicted_target_pose, grasp_idx)
            else:
                grasp_idx, grasp_planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv, grasp_switched \
                    = self.plan_grasp(predicted_target_pose, grasp_idx)
            dynamic_grasp_time += grasp_planning_time
            if planned_grasp_jv is None or planned_pre_grasp_jv is None:
                self.step(grasp_planning_time, None, None)
                continue
            self.step(grasp_planning_time, None, None)
            pu.create_arrow_marker(planned_pre_grasp, color_index=grasp_idx)

            # plan a motion
            distance = np.linalg.norm(np.array(self.robot.get_eef_pose()[0]) - np.array(planned_pre_grasp[0]))
            distance_travelled = np.linalg.norm(np.array(current_target_pose[0]) - np.array(
                last_motion_plan_success_pos)) if initial_motion_plan_success else 0
            if self.check_lazy_plan(distance, grasp_switched, distance_travelled):
                # print("lazy plan")
                continue
            motion_planning_time, plan = self.plan_arm_motion(planned_pre_grasp_jv)
            dynamic_grasp_time += motion_planning_time
            if plan is None:
                self.step(motion_planning_time, None, None)
                continue
            self.step(motion_planning_time, plan, None)
            last_motion_plan_success_pos = current_target_pose[0]
            initial_motion_plan_success = True

            # check can grasp or not
            can_grasp = self.can_grasp(grasp_idx)
            can_grasp_old = self.robot.equal_conf(self.robot.get_arm_joint_values(), planned_pre_grasp_jv, tol=self.grasp_threshold)
            if can_grasp or can_grasp_old:
                if self.approach_prediction:
                    # one extra IK call, right now ignore the time because it is very small
                    predicted_target_pose, predicted_conveyor_pose = self.predict(self.approach_prediction_duration)
                    planned_grasp_in_object = pu.split_7d(self.grasps_eef[grasp_idx])
                    planned_grasp = gu.convert_grasp_in_object_to_world(predicted_target_pose, planned_grasp_in_object)
                    planned_grasp_jv = self.robot.get_arm_ik(planned_grasp, avoid_collisions=False,
                                                             arm_joint_values=self.robot.get_arm_joint_values())
                    if planned_grasp_jv is None:
                        print("the predicted approach motion is not reachable")
                        continue

                motion_planning_time, arm_motion_plan, gripper_motion_plan = self.plan_approach_motion(planned_grasp_jv,
                                                                                                       self.approach_prediction_duration)
                dynamic_grasp_time += motion_planning_time
                self.execute_appraoch_and_grasp(arm_motion_plan, gripper_motion_plan)
                self.execute_lift()
                return self.check_success(), grasp_idx, dynamic_grasp_time
        return False, None, dynamic_grasp_time

    def plan_approach_motion(self, grasp_jv, prediction_duration):
        """ Plan the discretized approach motion for both arm and gripper """
        # no need to prediction in the old trajectory because plan simple takes about 0.001
        predicted_period = 0
        start_time = time.time()

        if self.robot.arm_discretized_plan is not None:
            future_target_index = min(int(predicted_period * 240 + self.robot.arm_wp_target_index),
                                      len(self.robot.arm_discretized_plan) - 1)
            if future_target_index == -1:
                # catch error
                print(self.robot.arm_discretized_plan)
                import ipdb
                ipdb.set_trace()
            start_joint_values = self.robot.arm_discretized_plan[future_target_index]
            arm_discretized_plan = self.robot.plan_arm_joint_values_simple(grasp_jv,
                                                                           start_joint_values=start_joint_values,
                                                                           duration=prediction_duration)
        else:
            arm_discretized_plan = self.robot.plan_arm_joint_values_simple(grasp_jv, duration=prediction_duration)

        # there is no gripper discretized plan
        gripper_discretized_plan = self.robot.plan_gripper_joint_values(self.robot.CLOSED_POSITION,
                                                                        duration=prediction_duration)

        planning_time = time.time() - start_time
        print("Planning a motion takes {:.6f}".format(planning_time))
        return planning_time, arm_discretized_plan, gripper_discretized_plan

    def execute_appraoch_and_grasp(self, arm_plan, gripper_plan):
        """ modify the arm and gripper plans according to close delay and execute it """
        arm_len = len(arm_plan)
        num_delay_steps = int(arm_len * self.close_delay)
        gripper_len = len(gripper_plan)
        final_len = max(arm_len, gripper_len + num_delay_steps)

        arm_plan = np.vstack(
            (arm_plan, np.tile(arm_plan[-1], (final_len - arm_len, 1)))) if arm_len <= final_len else arm_plan
        gripper_plan = np.vstack((np.tile(gripper_plan[0], (num_delay_steps, 1)), gripper_plan))
        gripper_plan = np.vstack((gripper_plan, np.tile(gripper_plan[-1], (final_len - len(gripper_plan), 1)))) if len(
            gripper_plan) <= final_len else gripper_plan
        assert len(arm_plan) == len(gripper_plan)
        for arm_wp, gripper_wp in zip(arm_plan, gripper_plan):
            self.robot.control_arm_joints(arm_wp)
            self.robot.control_gripper_joints(gripper_wp)
            self.conveyor.step()
            p.stepSimulation()
            self.world_steps += 1
            if self.world_steps % self.pose_steps == 0:
                self.motion_predictor_kf.update(pu.get_body_pose(self.target))

    def execute_lift(self):
        plan, fraction = self.robot.plan_cartesian_control(z=0.07)
        if fraction != 1.0:
            print('fraction {} not 1'.format(fraction))
        if plan is not None:
            self.robot.execute_arm_plan(plan, self.realtime)

    def get_ik_error(self, eef_pose, ik_result, coeff=0.4):

        # fk_result = self.robot.get_arm_fk(ik_result)
        fk_result = self.robot.get_arm_fk_pybullet(ik_result)

        trans_dist = np.linalg.norm(np.array(eef_pose[0]) - np.array(fk_result[0]))
        ang_dist = np.abs(np.dot(np.array(eef_pose[1]), np.array(fk_result[1])))

        return (1 - coeff) * trans_dist + coeff * ang_dist

    def get_iks_pregrasp_and_grasp_approximate(self, query_grasp_idx, target_pose):
        planned_pre_grasp_in_object = pu.split_7d(self.pre_grasps_eef[query_grasp_idx])
        planned_pre_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_pre_grasp_in_object)
        planned_pre_grasp_jv = self.robot.get_arm_ik_pybullet(planned_pre_grasp)

        planned_grasp_in_object = pu.split_7d(self.grasps_eef[query_grasp_idx])
        planned_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_grasp_in_object)
        planned_grasp_jv = self.robot.get_arm_ik_pybullet(planned_grasp, arm_joint_values=planned_pre_grasp_jv)

        return planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

    def plan_grasp_baseline(self, target_pose, old_grasp_idx):
        """ Plan a reachable pre_grasp and grasp pose"""
        # timing of the best machine
        ik_call_time = 0.01
        # rank_grasp_time = 0.135

        # optionally rank grasp based on reachability
        rank_grasp_time_start = time.time()
        grasp_order_idxs = self.rank_grasps(target_pose)
        rank_grasp_time = time.time() - rank_grasp_time_start
        print('rank grasp takes {}'.format(rank_grasp_time))

        planned_pre_grasps, planned_pre_grasp_jvs, planned_grasps, planned_grasp_jvs = [], [], [], []
        grasp_order_idxs = grasp_order_idxs[:self.max_check]
        if old_grasp_idx is not None:
            np.append(grasp_order_idxs, old_grasp_idx)  # always add old grasp index
        for i, grasp_idx in enumerate(grasp_order_idxs):
            planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv = self.get_iks_pregrasp_and_grasp_approximate(
                grasp_idx, target_pose)
            map(lambda x, y: x.append(y),
                [planned_pre_grasps, planned_pre_grasp_jvs, planned_grasps, planned_grasp_jvs],
                [planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv])

        pregrasp_ik_error = [self.get_ik_error(ee_pose, ik) for ee_pose, ik in
                             zip(planned_pre_grasps, planned_pre_grasp_jvs)]
        grasp_ik_error = [self.get_ik_error(ee_pose, ik) for ee_pose, ik in zip(planned_grasps, planned_grasp_jvs)]
        ik_error_sum = np.array(pregrasp_ik_error) + np.array(grasp_ik_error)
        min_error_idx = np.argmin(ik_error_sum)
        margin = 0.02  # TODO: check if this margin makes sense
        if old_grasp_idx and ik_error_sum[-1] < ik_error_sum[min_error_idx] + margin:   # -1: old grasp index is last in list, keep old grasp if error is not far from min
            grasp_idx = -1
        else:
            grasp_idx = min_error_idx
        grasp_switched = (grasp_idx != old_grasp_idx)

        num_ik_called = 2 * len(planned_pre_grasps)
        planning_time = rank_grasp_time + num_ik_called * ik_call_time
        # print("Planning a grasp takes {:.6f}".format(planning_time))

        return grasp_order_idxs[min_error_idx], planning_time, num_ik_called, planned_pre_grasps[grasp_idx], \
               planned_pre_grasp_jvs[grasp_idx], planned_grasps[grasp_idx], planned_grasp_jvs[grasp_idx], grasp_switched

    def get_iks_pregrasp_and_grasp(self, query_grasp_idx, target_pose):
        """ return 1 or 2 ik called; if successful, then planned_grasp_jv is not None """
        # the actual IK call maximum possible time is 0.1s
        num_ik_called = 0
        planned_pre_grasp_in_object = pu.split_7d(self.pre_grasps_eef[query_grasp_idx])
        planned_pre_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_pre_grasp_in_object)
        planned_pre_grasp_jv = self.robot.get_arm_ik(planned_pre_grasp, timeout=0.02, restarts=5)
        num_ik_called += 1

        planned_grasp, planned_grasp_jv = None, None
        if planned_pre_grasp_jv is not None:
            planned_grasp_in_object = pu.split_7d(self.grasps_eef[query_grasp_idx])
            planned_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_grasp_in_object)
            planned_grasp_jv = self.robot.get_arm_ik(planned_grasp, avoid_collisions=False,
                                                     arm_joint_values=planned_pre_grasp_jv,
                                                     timeout=0.02, restarts=5)
            num_ik_called += 1
        return num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

    def rank_grasps(self, target_pose):
        pre_grasps_link6_ref_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
                                         self.pre_grasps_link6_ref]

        if self.disable_reachability:
            grasp_order_idxs = np.random.permutation(np.arange(len(pre_grasps_link6_ref_in_world)))
        else:
            sdf_values = gu.get_reachability_of_grasps_pose_2d(pre_grasps_link6_ref_in_world,
                                                               self.sdf_reachability_space,
                                                               self.mins,
                                                               self.step_size,
                                                               self.dims)
            grasp_order_idxs = np.argsort(sdf_values)[::-1]

        # grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
        #                        self.grasps_eef]
        # gu.visualize_grasps_with_reachability(grasps_eef_in_world, sdf_values)
        # gu.visualize_grasp_with_reachability(planned_grasp, sdf_values[grasp_order_idxs[0]],
        #                                      maximum=max(sdf_values), minimum=min(sdf_values))

        # pick top 10 reachable grasp for motion aware quality ranking
        if self.use_motion_aware:
            # max_check must be smaller than this number
            num_motion_aware_grasps = 10
            most_reachable_grasps_indices = grasp_order_idxs[:num_motion_aware_grasps]
            if self.conveyor.direction == 1:
                conveyor_angle_in_world = self.conveyor.theta + 90
            elif self.conveyor.direction == -1:
                conveyor_angle_in_world = self.conveyor.theta - 90
            else:
                raise TypeError
            # visualize target pose
            # pu.create_frame_marker(target_pose)
            target_angle_in_world = degrees(pu.get_euler_from_quaternion(target_pose[1])[2])
            conveyor_angle_in_object = conveyor_angle_in_world - target_angle_in_world
            speed = self.conveyor.speed

            most_reachable_grasps_eef = [self.grasps_eef[i] for i in most_reachable_grasps_indices]
            most_reachable_pre_grasps_eef = [self.pre_grasps_eef[i] for i in most_reachable_grasps_indices]
            # start = time.time()
            motion_aware_qualities = self.get_motion_aware_qualities(most_reachable_grasps_eef,
                                                                     most_reachable_pre_grasps_eef,
                                                                     radians(conveyor_angle_in_object),
                                                                     speed)
            # print(time.time() - start)
            grasp_order_idxs = [x for _, x in sorted(zip(motion_aware_qualities, most_reachable_grasps_indices))]

            # visualization
            # all_motion_aware_qualities = self.get_motion_aware_qualities(self.grasps_eef,
            #                                                          self.pre_grasps_eef,
            #                                                          radians(conveyor_angle_in_object),
            #                                                          speed)
            # grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
            #                        self.grasps_eef]
            # gu.visualize_grasps_with_reachability(grasps_eef_in_world, all_motion_aware_qualities)
            # gu.visualize_grasp_with_reachability(grasps_eef_in_world[grasp_order_idxs[0]], sdf_values[grasp_order_idxs[0]],
            #                                      maximum=max(sdf_values), minimum=min(sdf_values))

        return grasp_order_idxs

    def get_motion_aware_qualities(self, grasps_eef, pre_grasps_eef, angle, speed):
        """

        :param grasps_eef: a list of grasps in eef reference frame
        :param pre_grasps_eef: a list of pre grasps in eef reference frame
        :param angle: angle in radians
        :param speed: speed in m/s
        :return:
        """
        qualities = [self.compute_motion_aware_quality(g, pg, angle, speed) for g, pg in zip(grasps_eef, pre_grasps_eef)]
        return qualities

    def compute_motion_aware_quality(self, grasp_pose_7d_in_object, pre_grasp_pose_7d_in_object, angle, speed):
        """

        :param grasp_pose_7d_in_object: 7d grasp pose in object frame and uses eef reference frame
        :param pre_grasp_pose_7d_in_object: 7d pre grasp pose in object frame and uses eef reference frame
        :return:
        """
        x = torch.tensor(list(grasp_pose_7d_in_object) + list(pre_grasp_pose_7d_in_object) + [angle] + [speed])
        logits = self.motion_aware_network(x)
        probs = softmax(logits)
        quality = probs[1]
        return quality.item()

    def plan_grasp(self, target_pose, old_grasp_idx):
        """ Plan a reachable pre_grasp and grasp pose"""
        # timing of the best machine
        ik_call_time = 0.01

        num_ik_called = 0
        grasp_idx = None
        grasp_switched = False

        # if an old grasp index is provided
        if old_grasp_idx is not None:
            _num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv = self.get_iks_pregrasp_and_grasp(
                old_grasp_idx, target_pose)
            num_ik_called += _num_ik_called
            if planned_grasp_jv is not None:
                planning_time = num_ik_called * ik_call_time
                return old_grasp_idx, planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv, grasp_switched

        # if an old grasp index is not provided or the old grasp is not reachable any more
        rank_grasp_time_start = time.time()
        grasp_order_idxs = self.rank_grasps(target_pose)
        actual_rank_grasp_time = time.time() - rank_grasp_time_start
        rank_grasp_time = actual_rank_grasp_time if self.fix_grasp_ranking_time is None else self.fix_grasp_ranking_time
        print('Rank grasp actually takes {:.6f}, fixed grasp ranking time {:.6}'.format(actual_rank_grasp_time,
                                                                                        self.fix_grasp_ranking_time))

        for i, grasp_idx in enumerate(grasp_order_idxs):
            if i == self.max_check:
                break
            _num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv = self.get_iks_pregrasp_and_grasp(
                grasp_idx, target_pose)
            num_ik_called += _num_ik_called
            if planned_grasp_jv is not None:
                grasp_switched = (grasp_idx != old_grasp_idx)
                break

        if planned_pre_grasp_jv is None:
            print('pre grasp planning fails')
            grasp_idx = None
        if planned_pre_grasp_jv is not None and planned_grasp_jv is None:
            print('pre grasp planning succeeds but grasp planning fails')
            grasp_idx = None
        planning_time = rank_grasp_time + num_ik_called * ik_call_time
        print("Planning a grasp takes {:.6f}".format(planning_time))
        return grasp_idx, planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv, grasp_switched

    def plan_arm_motion(self, grasp_jv):
        """ plan a discretized motion for the arm """
        # whether we should have a fixed planning time
        predicted_period = 0.25 if self.fix_motion_planning_time is None else self.fix_motion_planning_time
        start_time = time.time()

        if self.robot.arm_discretized_plan is not None:
            future_target_index = min(int(predicted_period * 240 + self.robot.arm_wp_target_index),
                                      len(self.robot.arm_discretized_plan) - 1)
            start_joint_values = self.robot.arm_discretized_plan[future_target_index]
            start_joint_velocities = None
            if self.use_previous_jv:
                next_joint_values = self.robot.arm_discretized_plan[
                    min(future_target_index + 1, len(self.robot.arm_discretized_plan) - 1)]
                start_joint_velocities = (next_joint_values - start_joint_values) / (
                        1. / 240)  # TODO: confirm that getting joint velocity this way is right
            previous_discretized_plan = self.robot.arm_discretized_plan[
                                        future_target_index:] if self.use_seed_trajectory else None
            arm_discretized_plan = self.robot.plan_arm_joint_values(grasp_jv, start_joint_values=start_joint_values,
                                                                    previous_discretized_plan=previous_discretized_plan,
                                                                    start_joint_velocities=start_joint_velocities)
        else:
            arm_discretized_plan = self.robot.plan_arm_joint_values(grasp_jv)

        actual_planning_time = time.time() - start_time
        planning_time = actual_planning_time if self.fix_motion_planning_time is None else self.fix_motion_planning_time
        print("Planning a motion actually takes {:.6f}, fixed motion planning time {:.6}".format(actual_planning_time,
                                                                                                 self.fix_motion_planning_time))
        return planning_time, arm_discretized_plan

    def sample_target_location(self):
        r = np.random.uniform(low=self.distance_low, high=self.distance_high)
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        x = r * cos(theta)
        y = r * sin(theta)
        distance = np.linalg.norm(np.array([x, y]) - np.array(self.robot_initial_pose[0][:2]))
        z = self.target_initial_pose[0][2]
        angle = np.random.uniform(-pi, pi)
        pos = [x, y, z]
        orientation = p.getQuaternionFromEuler([0, 0, angle])
        return [pos, orientation], distance

    @staticmethod
    def sample_target_angle():
        """ return quaternion """
        angle = np.random.uniform(-pi, pi)
        orientation = p.getQuaternionFromEuler([0, 0, angle])
        return list(orientation)

    def sample_convey_linear_motion(self, dist=None, theta=None, length=None, direction=None):
        """ theta is in degrees """
        if dist is None:
            dist = np.random.uniform(low=self.distance_low, high=self.distance_high)
        if theta is None:
            theta = np.random.uniform(low=0, high=360)
        if length is None:
            length = 1.0
        if direction is None:
            direction = random.sample([-1, 1], 1)[0]
        return dist, theta, length, direction

    def sample_convey_circular_motion(self, dist=None, theta=None, length=None, direction=None):
        """
        theta is in degrees

        dist: the distance from the robot,
        theta: the angle of the starting position,
        length: the length of the trajectory
        direction: 1 is counter clockwise, -1 is clockwise
        """
        # this is effectively the only difference
        if dist is None:
            dist = np.random.uniform(low=self.circular_distance_low, high=self.circular_distance_high)
        if theta is None:
            theta = np.random.uniform(low=0, high=360)
        if length is None:
            length = 1.0
        if direction is None:
            direction = random.sample([-1, 1], 1)[0]
        return dist, theta, length, direction


    def get_obstacles_regions(self, distance, theta, length, visualize_region=True):
        region_length = (length - 2 * self.distance_between_region) / 3
        theta = theta - 90
        translation = np.array([[1, 0, 0],
                                [0, 1, distance],
                                [0, 0, 1]])
        rotation = np.array([[cos(radians(theta)), -sin(radians(theta)), 0],
                             [sin(radians(theta)), cos(radians(theta)), 0],
                             [0, 0, 1]])
        transform_matrix = rotation.dot(translation)

        # there are 6 regions
        regions = [[(-length / 2.0, self.obstacle_distance_low),
                    (-length / 2.0, self.obstacle_distance_high),
                    (-length / 2.0 + region_length, self.obstacle_distance_high),
                    (-length / 2.0 + region_length, self.obstacle_distance_low)],

                   [(-length / 2.0 + region_length + self.distance_between_region, self.obstacle_distance_low),
                    (-length / 2.0 + region_length + self.distance_between_region, self.obstacle_distance_high),
                    (-length / 2.0 + 2 * region_length + self.distance_between_region, self.obstacle_distance_high),
                    (-length / 2.0 + 2 * region_length + self.distance_between_region, self.obstacle_distance_low)],

                   [(-length / 2.0 + 2 * region_length + 2 * self.distance_between_region, self.obstacle_distance_low),
                    (-length / 2.0 + 2 * region_length + 2 * self.distance_between_region, self.obstacle_distance_high),
                    (-length / 2.0 + 3 * region_length + 2 * self.distance_between_region, self.obstacle_distance_high),
                    (-length / 2.0 + 3 * region_length + 2 * self.distance_between_region, self.obstacle_distance_low)],

                   [(-length / 2.0 + 2 * region_length + 2 * self.distance_between_region, -self.obstacle_distance_low),
                    (-length / 2.0 + 2 * region_length + 2 * self.distance_between_region, -self.obstacle_distance_high),
                    (-length / 2.0 + 3 * region_length + 2 * self.distance_between_region, -self.obstacle_distance_high),
                    (-length / 2.0 + 3 * region_length + 2 * self.distance_between_region, -self.obstacle_distance_low)],

                   [(-length / 2.0 + region_length + self.distance_between_region, -self.obstacle_distance_low),
                    (-length / 2.0 + region_length + self.distance_between_region, -self.obstacle_distance_high),
                    (-length / 2.0 + 2 * region_length + self.distance_between_region, -self.obstacle_distance_high),
                    (-length / 2.0 + 2 * region_length + self.distance_between_region, -self.obstacle_distance_low)],

                   [(-length / 2.0, -self.obstacle_distance_low),
                    (-length / 2.0, -self.obstacle_distance_high),
                    (-length / 2.0 + region_length, -self.obstacle_distance_high),
                    (-length / 2.0 + region_length, -self.obstacle_distance_low)]]

        regions_transformed = []
        for r in regions:
            r_transformed = []
            for point in r:
                r_transformed.append(tuple(transform_matrix.dot(np.array(point + (1,)))[:2]))
            regions_transformed.append(r_transformed)

        # visualize
        if visualize_region:
            v_height = 0.01
            for r in regions_transformed:
                lines = zip(r, r[1:] + [r[0]])
                for (p1, p2) in lines:
                    pu.draw_line(p1 + (v_height,), p2 + (v_height,), rgb_color=(0, 1, 0))

        # create original polygons
        polygons = [Polygon(r) for r in regions]
        return polygons, transform_matrix

    def load_obstacles_collision_free(self, distance, theta, length):
        polygons, transform_matrix = self.get_obstacles_regions(distance, theta, length)

        poses = []
        obstacles = []
        choices = random.choice(list(combinations(range(6), len(self.obstacle_names))))
        for choice, urdf, extents, z in zip(choices, self.obstacle_urdfs, self.obstacle_extentss, self.obstacle_zs):
            position_xy = mu.random_point_in_polygon(polygons[choice])
            position_xy = tuple(transform_matrix.dot(np.array(position_xy + (1,)))[:2])
            pose = [list(position_xy) + [z], self.sample_target_angle()]
            poses.append(pose)
            obstacles.append(p.loadURDF(urdf, pose[0], pose[1]))
        return obstacles

    def load_obstacles_at_poses(self, poses):
        obstacles = []
        for pose_flattened, urdf in zip(poses, self.obstacle_urdfs):
            pose = pu.split_7d(pose_flattened)
            obstacles.append(p.loadURDF(urdf, pose[0], pose[1]))
        return obstacles

    def check_done(self):
        done = False
        if self.conveyor.wp_target_index == len(self.conveyor.discretized_trajectory):
            # conveyor trajectory finishes
            done = True
        if pu.get_body_pose(self.target)[0][2] < self.target_initial_pose[0][2] - 0.03:
            # target knocked down
            done = True
        return done

    def check_lazy_plan(self, distance, grasp_switched, distance_travelled):
        """ check whether we should do lazy plan """
        do_lazy_plan = distance > self.lazy_threshold and \
                       distance_travelled < self.distance_travelled_threshold and \
                       self.robot.arm_discretized_plan is not None and \
                       self.robot.arm_wp_target_index != len(self.robot.arm_discretized_plan) and \
                       not grasp_switched
        return do_lazy_plan

    def calculate_prediction_time(self, distance):
        if distance is None:
            # print('large')
            prediction_time = 2
        else:
            if self.small_prediction_threshold < distance <= self.large_prediction_threshold:
                # print('medium')
                prediction_time = 1
            elif distance <= self.small_prediction_threshold:
                # print('small')
                prediction_time = 0
            else:
                # print('large')
                prediction_time = 2
        return prediction_time


class Conveyor:
    def __init__(self, initial_pose, urdf_path):
        self.initial_pose = initial_pose
        self.urdf_path = urdf_path
        self.id = p.loadURDF(self.urdf_path, initial_pose[0], initial_pose[1])

        self.cid = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                      childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0], childFramePosition=initial_pose[0],
                                      childFrameOrientation=initial_pose[1])

        # motion related
        self.start_pose = None
        self.target_pose = None
        self.discretized_trajectory = None
        self.wp_target_index = 0
        self.distance = None
        self.theta = None
        self.length = None
        self.direction = None
        self.speed = None

    def set_pose(self, pose):
        pu.set_pose(self.id, pose)
        self.control_pose(pose)

    def get_pose(self):
        return pu.get_body_pose(self.id)

    def control_pose(self, pose):
        p.changeConstraint(self.cid, jointChildPivot=pose[0], jointChildFrameOrientation=pose[1])

    def step(self):
        if self.discretized_trajectory is None or self.wp_target_index == len(self.discretized_trajectory):
            pass
        else:
            self.control_pose(self.discretized_trajectory[self.wp_target_index])
            self.wp_target_index += 1

    def initialize_linear_motion(self, dist, theta, length, direction, speed):
        """
        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 360)
        :param length: the length of the motion
        :param direction: the direction of the motion
            1: from smaller theta to larger theta
            -1: from larger theta to smaller theta
        :param speed: the speed of the conveyor
        """
        self.distance = float(dist)
        self.theta = float(theta)
        self.length = float(length)
        self.direction = float(direction)
        self.speed = float(speed)
        # uses the z value and orientation of the current pose
        z = self.get_pose()[0][-1]
        orientation = self.get_pose()[1]
        # compute start xy and end xy
        new_dist = sqrt(dist ** 2 + (length / 2.0) ** 2)
        delta_theta = atan((length / 2.0) / dist)

        theta_large = radians(self.theta) + delta_theta
        theta_small = radians(self.theta) - delta_theta

        if direction == -1:
            start_xy = [new_dist * cos(theta_large), new_dist * sin(theta_large)]
            target_xy = [new_dist * cos(theta_small), new_dist * sin(theta_small)]
        elif direction == 1:
            target_xy = [new_dist * cos(theta_large), new_dist * sin(theta_large)]
            start_xy = [new_dist * cos(theta_small), new_dist * sin(theta_small)]
        else:
            raise ValueError('direction must be in {-1, 1}')
        start_position = start_xy + [z]
        target_position = target_xy + [z]

        self.start_pose = [start_position, orientation]
        self.target_pose = [target_position, orientation]

        num_steps = int(self.length / self.speed * 240)
        position_trajectory = np.linspace(start_position, target_position, num_steps)
        self.discretized_trajectory = [[list(p), orientation] for p in position_trajectory]
        self.wp_target_index = 1

    def initialize_circular_motion(self, dist, theta, length, direction, speed):
        """
        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 360)
        :param length: the length of the motion
        :param direction: the direction of the motion
            1: counter clockwise
            -1: clockwise
        :param speed: the speed of the conveyor
        """
        self.distance = float(dist)
        self.theta = float(theta)
        self.length = float(length)
        self.direction = float(direction)
        self.speed = float(speed)
        # uses the z value and orientation of the current pose
        z = self.get_pose()[0][-1]
        orientation = self.get_pose()[1]

        # calculate waypoints
        num_points = int(self.length / self.speed) * 240
        delta_angle = self.length / self.distance
        angles = np.linspace(radians(theta), radians(theta)+delta_angle, num_points)
        if direction == -1:
            angles = angles[::-1]

        self.discretized_trajectory = [[(cos(ang) * self.distance, sin(ang) * self.distance, z), orientation] for ang in angles]
        self.wp_target_index = 1

        self.start_pose = self.discretized_trajectory[0]
        self.target_pose = self.discretized_trajectory[-1]

    def initialize_linear_motion_v2(self, angle, speed, distance, start_pose=None):
        """
        Initialize a motion using the start pose as initial pose, in the direction of the angle.

        :param angle: the angle of the motion direction in the conveyor frame, in degrees
        :param speed: the speed of the motion
        """
        start_pose_in_world = conveyor_pose = self.get_pose() if start_pose is None else start_pose
        start_pose_in_conveyor = [[0, 0, 0], [0, 0, 0, 1]]

        target_x = cos(radians(angle)) * distance
        target_y = sin(radians(angle)) * distance
        target_pose_in_conveyor = [[target_x, target_y, 0], [0, 0, 0, 1]]
        target_pose_in_world = tfc.toMatrix(tfc.fromTf(conveyor_pose)).dot(
            tfc.toMatrix(tfc.fromTf(target_pose_in_conveyor)))
        target_pose_in_world = tfc.toTf(tfc.fromMatrix(target_pose_in_world))
        target_pose_in_world = [list(target_pose_in_world[0]), list(target_pose_in_world[1])]

        start_pose = start_pose_in_world
        target_pose = target_pose_in_world

        num_steps = int(distance / speed * 240)
        position_trajectory = np.linspace(start_pose[0], target_pose[0], num_steps)
        self.discretized_trajectory = [[list(p), start_pose[1]] for p in position_trajectory]
        self.wp_target_index = 1
        return start_pose, target_pose

    def clear_motion(self):
        self.start_pose = None
        self.target_pose = None
        self.discretized_trajectory = None
        self.wp_target_index = 0
        self.distance = None
        self.theta = None
        self.length = None
        self.direction = None

    def predict(self, duration):
        # predict the ground truth future pose of the conveyor
        num_predicted_steps = int(duration * 240)
        predicted_step_index = self.wp_target_index - 1 + num_predicted_steps
        if predicted_step_index < len(self.discretized_trajectory):
            return self.discretized_trajectory[predicted_step_index]
        else:
            return self.discretized_trajectory[-1]


class MotionPredictorKF:
    def __init__(self, time_step):
        # the predictor takes a pose estimation once every time step
        self.time_step = time_step
        self.target_pose = None
        self.kf = None
        self.initialized = False

    def initialize_predictor(self, initial_pose):
        self.target_pose = initial_pose
        x0 = np.zeros(9)
        x0[:3] = initial_pose[0]
        # x0[3] = 0.03
        x0 = x0[:, None]
        self.kf = create_kalman_filter(x0=x0)
        self.initialized = True

    def reset_predictor(self):
        self.target_pose = None
        self.kf = None

    def update(self, current_pose):
        # TODO quaternion is not considered yet
        if not self.initialized:
            raise ValueError("predictor not initialized!")
        self.target_pose = current_pose
        current_position = current_pose[0]
        self.kf.predict(dt=self.time_step)
        self.kf.update(np.array(current_position)[:, None])

    def predict(self, duration):
        """ return just a predicted pose """
        if not self.initialized:
            raise ValueError("predictor not initialized!")
        # print("current position: {}".format(self.target_pose[0]))
        future_estimate = np.dot(self.kf.H, self.kf.predict(dt=duration, predict_only=True))
        future_position = list(np.squeeze(future_estimate))
        # print("future position: {}\n".format(future_position))
        future_orientation = self.target_pose[1]
        return [future_position, future_orientation]
