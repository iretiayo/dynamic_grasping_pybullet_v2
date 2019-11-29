import os
import numpy as np
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
from math import pi, cos, sin, sqrt, atan


class DynamicGraspingWorld:
    def __init__(self,
                 target_name,
                 target_initial_pose,
                 robot_initial_pose,
                 robot_initial_state,
                 conveyor_initial_pose,
                 robot_urdf,
                 conveyor_urdf,
                 conveyor_speed,
                 target_urdf,
                 target_mesh_file_path,
                 grasp_database_path,
                 reachability_data_dir,
                 realtime,
                 max_check,
                 disable_reachability,
                 back_off):
        self.target_name = target_name
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
        self.realtime = realtime
        self.max_check = max_check
        self.back_off = back_off
        self.disable_reachability = disable_reachability

        self.x_low = -0.4
        self.y_low = -0.4
        self.x_high = 0.4
        self.y_high = 0.4
        self.distance_low = 0.1
        self.distance_high = 0.4

        self.grasp_database_path = grasp_database_path
        self.grasps_eef = np.load(os.path.join(self.grasp_database_path, self.target_name, 'grasps_eef.npy'))
        self.grasps_link6_ref = np.load(
            os.path.join(self.grasp_database_path, self.target_name, 'grasps_link6_ref.npy'))
        self.grasps_link6_com = np.load(
            os.path.join(self.grasp_database_path, self.target_name, 'grasps_link6_com.npy'))
        self.pre_grasps_eef = np.load(
            os.path.join(self.grasp_database_path, self.target_name, 'pre_grasps_eef_' + str(self.back_off) + '.npy'))
        self.pre_grasps_link6_ref = np.load(
            os.path.join(self.grasp_database_path, self.target_name,
                         'pre_grasps_link6_ref_' + str(self.back_off) + '.npy'))

        self.reachability_data_dir = reachability_data_dir
        self.sdf_reachability_space, self.mins, self.step_size, self.dims = gu.get_reachability_space(
            self.reachability_data_dir)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])
        self.robot = MicoController(self.robot_initial_pose, self.robot_initial_state, self.robot_urdf)
        self.conveyor = Conveyor(self.conveyor_initial_pose, self.conveyor_urdf, self.conveyor_speed)

        self.target_pose_pub = rospy.Publisher('target_pose', PoseStamped, queue_size=1)
        self.conveyor_pose_pub = rospy.Publisher('conveyor_pose', PoseStamped, queue_size=1)
        self.target_mesh_file_path_pub = rospy.Publisher('target_mesh', String, queue_size=1)

        update_scene_thread = threading.Thread(target=self.update_scene_threading)
        update_scene_thread.daemon = True
        update_scene_thread.start()

    def update_scene_threading(self):
        r = rospy.Rate(30)
        while True:
            target_pose = pu.get_body_pose(self.target)
            conveyor_pose = pu.get_body_pose(self.conveyor.id)
            self.target_pose_pub.publish(gu.list_2_ps(target_pose))
            self.conveyor_pose_pub.publish(gu.list_2_ps(conveyor_pose))
            self.target_mesh_file_path_pub.publish(self.target_mesh_file_path)
            r.sleep()
            # target_pose = pu.get_body_pose(self.target)
            # conveyor_pose = pu.get_body_pose(self.conveyor)
            # self.robot.scene.add_mesh(self.target_name, gu.list_2_ps(target_pose), self.target_mesh_file_path)
            # self.robot.scene.add_box('conveyor', gu.list_2_ps(conveyor_pose), size=(.1, .1, .02))

    def reset(self, mode):
        """
        mode:
            initial:
            static_random: reset the target to a random pose, not moving
            dynamic_linear: initialize the conveyor with a linear motion
            dynamic_circular: initialize the conveyor with a circular motion
            hand_over: TODO
        """
        if mode == 'initial':
            pu.remove_all_markers()
            target_pose, distance = self.target_initial_pose, self.initial_distance
            conveyor_pose = [[target_pose[0][0], target_pose[0][1], 0.01],
                             [0, 0, 0, 1]] if target_pose is not None else self.conveyor_initial_pose
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.reset(conveyor_pose)
            self.robot.reset()
            pu.step(2)
            return target_pose, distance

        elif mode == 'static_random':
            pu.remove_all_markers()
            target_pose, distance = self.sample_target_location()
            conveyor_pose = [[target_pose[0][0], target_pose[0][1], 0.01],
                             [0, 0, 0, 1]] if target_pose is not None else self.conveyor_initial_pose
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.reset(conveyor_pose)
            self.robot.reset()
            pu.step(2)
            return target_pose, distance

        elif mode == 'dynamic_linear':
            pu.remove_all_markers()
            target_pose, distance = self.sample_target_location()
            conveyor_pose = [[target_pose[0][0], target_pose[0][1], 0.01],
                             [0, 0, 0, 1]] if target_pose is not None else self.conveyor_initial_pose
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.reset(conveyor_pose)
            self.robot.reset()
            pu.step(2)

            conveyor_target_position, conveyor_dist = self.sample_conveyor_target_position()
            self.conveyor.initialize_trajectory(conveyor_target_position)
            pu.draw_line(self.conveyor.start_pose[0], conveyor_target_position)
            return target_pose, distance

        elif mode == 'dynamic_circular':
            raise NotImplementedError
        elif mode == 'hand_over':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def step(self, freeze_time, arm_motion_plan, gripper_motion_plan):
        for i in range(int(freeze_time * 240)):
            # step the robot
            self.robot.step()
            self.conveyor.step()
            # step python
            p.stepSimulation()
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
        grasp_idx, grasp_planning_time, num_ik_called, pre_grasp, pre_grasp_jv, grasp, grasp_jv = self.plan_grasp(
            predicted_pose, None)
        if grasp_jv is None or pre_grasp_jv is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, "no reachable grasp is found"

        # planning motion
        motion_planning_time, plan = self.plan_arm_motion(pre_grasp_jv)
        if plan is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, "no motion found to the planned pre grasp jv"

        # move
        self.robot.execute_arm_plan(plan, self.realtime)
        grasp_attempted = True
        pre_grasp_reached = self.robot.equal_conf(self.robot.get_arm_joint_values(), pre_grasp_jv, tol=0.01)

        # print('self')
        # print(self.robot.get_arm_joint_values())
        # print('pre_grasp_jv')
        # print(pre_grasp_jv)
        # print('grasp_jv')
        # print(grasp_jv)

        # approach
        plan = self.robot.plan_arm_joint_values_simple(grasp_jv)
        self.robot.execute_arm_plan(plan, self.realtime)
        grasp_reachaed = self.robot.equal_conf(self.robot.get_arm_joint_values(), grasp_jv, tol=0.01)
        # plan, fraction = self.robot.plan_cartesian_control(z=0.05, frame='eef')
        # self.robot.execute_plan(plan, self.realtime)
        # print(fraction)

        # close and lift
        self.robot.close_gripper(self.realtime)
        plan, fraction = self.robot.plan_cartesian_control(z=0.07)
        if fraction != 1.0:
            comment = "lift fration {} is not 1.0".format(fraction)
        self.robot.execute_arm_plan(plan, self.realtime)
        success = self.check_success()
        pu.remove_all_markers()
        return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, comment

    def check_success(self):
        if pu.get_body_pose(self.target)[0][2] >= self.target_initial_pose[0][2] + 0.03:
            return True
        else:
            return False

    def dynamic_grasp(self, grasp_threshold=0.01, close_delay=2):
        success = False
        grasp_idx = None
        while True:
            target_pose = pu.get_body_pose(self.target)
            predicted_pose = target_pose

            # plan a grasp
            grasp_idx, grasp_planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv \
                = self.plan_grasp(predicted_pose, grasp_idx)
            if planned_grasp_jv is None or planned_pre_grasp_jv is None:
                # return success, grasp_idx, grasp_attempted,  "no reachable grasp is found"
                self.step(grasp_planning_time, None, None)
                continue
            self.step(grasp_planning_time, None, None)

            # plan a motion
            motion_planning_time, plan = self.plan_arm_motion(planned_pre_grasp_jv)
            if plan is None:
                # return success, grasp_idx, grasp_attempted, "no motion found to the planned pre grasp jv"
                self.step(motion_planning_time, None, None)
                continue
            self.step(motion_planning_time, plan, None)

            # check can grasp or not
            if self.robot.equal_conf(self.robot.get_arm_joint_values(), planned_pre_grasp_jv, tol=grasp_threshold):
                motion_planning_time, arm_motion_plan, gripper_motion_plan = self.plan_approach_motion(planned_grasp_jv)
                self.execute_appraoch_and_grasp(arm_motion_plan, gripper_motion_plan, close_delay)
                self.execute_lift()
                return success

    def plan_approach_motion(self, grasp_jv):
        """ Plan the discretized approach motion for both arm and gripper """
        predicted_period = 0.2
        start_time = time.time()

        if self.robot.arm_discretized_plan is not None:
            future_target_index = min(int(predicted_period * 240 + self.robot.arm_wp_target_index),
                                      len(self.robot.arm_discretized_plan) - 1)
            start_joint_values = self.robot.arm_discretized_plan[future_target_index]
            arm_discretized_plan = self.robot.plan_arm_joint_values_simple(grasp_jv, start_joint_values=start_joint_values)
        else:
            arm_discretized_plan = self.robot.plan_arm_joint_values_simple(grasp_jv)

        # there is no gripper discretized plan
        gripper_discretized_plan = self.robot.plan_gripper_joint_values(self.robot.CLOSED_POSITION)

        planning_time = time.time() - start_time
        print("Planning a motion takes {:.6f}".format(planning_time))
        return planning_time, arm_discretized_plan, gripper_discretized_plan

    def execute_appraoch_and_grasp(self, arm_plan, gripper_plan, delay):
        num_delay_steps = int(delay * 240.0)
        arm_len = len(arm_plan)
        gripper_len = len(gripper_plan)
        final_len = max(arm_len, gripper_len+num_delay_steps)

        arm_plan = np.vstack((arm_plan, np.tile(arm_plan[-1], (final_len-arm_len, 1)))) if arm_len <= final_len else arm_plan
        gripper_plan = np.vstack((np.tile(gripper_plan[0], (num_delay_steps, 1)), gripper_plan))
        gripper_plan = np.vstack((gripper_plan, np.tile(gripper_plan[-1], (final_len-len(gripper_plan), 1)))) if len(gripper_plan) <= final_len else gripper_plan
        assert len(arm_plan) == len(gripper_plan)
        for arm_wp, gripper_wp in zip(arm_plan, gripper_plan):
            self.robot.control_arm_joints(arm_wp)
            self.robot.control_gripper_joints(gripper_wp)
            # step conveyor
            p.stepSimulation()

    def execute_lift(self):
        plan, fraction = self.robot.plan_cartesian_control(z=0.07)
        if fraction != 1.0:
            print('fraction {} not 1'.format(fraction))
        self.robot.execute_arm_plan(plan, self.realtime)

    def plan_grasp(self, target_pose, old_grasp_idx):
        """ Plan a reachable pre_grasp and grasp pose"""
        start_time = time.time()
        num_ik_called = 0
        grasp_idx = None
        planned_pre_grasp = None
        planned_pre_grasp_jv = None
        planned_grasp = None
        planned_grasp_jv = None

        # if an old grasp index is provided
        if old_grasp_idx is not None:
            planned_pre_grasp_in_object = pu.split_7d(self.pre_grasps_eef[old_grasp_idx])
            planned_pre_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_pre_grasp_in_object)
            planned_pre_grasp_jv = self.robot.get_arm_ik(planned_pre_grasp)
            if planned_pre_grasp_jv is not None:
                planned_grasp_in_object = pu.split_7d(self.grasps_eef[old_grasp_idx])
                planned_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_grasp_in_object)
                planned_grasp_jv = self.robot.get_arm_ik(planned_grasp, avoid_collisions=False,
                                                         arm_joint_values=planned_pre_grasp_jv)
                if planned_grasp_jv is not None:
                    planning_time = time.time() - start_time
                    print("Planning a grasp takes {:.6f}".format(planning_time))
                    return old_grasp_idx, planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

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
        for num_ik_called, grasp_idx in enumerate(grasp_order_idxs):
            if num_ik_called == self.max_check:
                break
            planned_pre_grasp_in_object = pu.split_7d(self.pre_grasps_eef[grasp_idx])
            planned_pre_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_pre_grasp_in_object)
            planned_pre_grasp_jv = self.robot.get_arm_ik(planned_pre_grasp)
            if planned_pre_grasp_jv is None:
                continue
            planned_grasp_in_object = pu.split_7d(self.grasps_eef[grasp_idx])
            planned_grasp = gu.convert_grasp_in_object_to_world(target_pose, planned_grasp_in_object)
            planned_grasp_jv = self.robot.get_arm_ik(planned_grasp, avoid_collisions=False,
                                                     arm_joint_values=planned_pre_grasp_jv)
            if planned_grasp_jv is None:
                continue
            num_ik_called += 1
            break

        # grasps_eef_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in
        #                        self.grasps_eef]
        # gu.visualize_grasps_with_reachability(grasps_eef_in_world, sdf_values)
        # gu.visualize_grasp_with_reachability(planned_grasp, sdf_values[grasp_order_idxs[0]],
        #                                      maximum=max(sdf_values), minimum=min(sdf_values))
        planning_time = time.time() - start_time
        print("Planning a grasp takes {:.6f}".format(planning_time))
        return grasp_idx, planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

    def plan_arm_motion(self, grasp_jv):
        """ plan a discretized motion for the arm """
        predicted_period = 0.2
        start_time = time.time()

        if self.robot.arm_discretized_plan is not None:
            future_target_index = min(int(predicted_period * 240 + self.robot.arm_wp_target_index),
                                      len(self.robot.arm_discretized_plan) - 1)
            start_joint_values = self.robot.arm_discretized_plan[future_target_index]
            arm_discretized_plan = self.robot.plan_arm_joint_values(grasp_jv, start_joint_values=start_joint_values)
        else:
            arm_discretized_plan = self.robot.plan_arm_joint_values(grasp_jv)
        planning_time = time.time() - start_time

        print("Planning a motion takes {:.6f}".format(planning_time))
        return planning_time, arm_discretized_plan

    def sample_target_location(self):
        x, y = np.random.uniform([-self.distance_high, -self.distance_high], [self.distance_high, self.distance_high])
        distance = np.linalg.norm(np.array([x, y]) - np.array(self.robot_initial_pose[0][:2]))
        while not self.distance_low <= distance <= self.distance_high:
            x, y = np.random.uniform([-self.distance_high, -self.distance_high],
                                     [self.distance_high, self.distance_high])
            distance = np.linalg.norm(np.array([x, y]) - np.array(self.robot_initial_pose[0][:2]))
        z = self.target_initial_pose[0][2]
        angle = np.random.uniform(-pi, pi)
        pos = [x, y, z]
        orientation = p.getQuaternionFromEuler([0, 0, angle])
        return [pos, orientation], distance

    def sample_conveyor_target_position(self):
        x, y = np.random.uniform([-self.distance_high, -self.distance_high], [self.distance_high, self.distance_high])
        distance = np.linalg.norm(np.array([x, y]) - np.array(self.robot_initial_pose[0][:2]))
        while not self.distance_low <= distance <= self.distance_high:
            x, y = np.random.uniform([-self.distance_high, -self.distance_high],
                                     [self.distance_high, self.distance_high])
            distance = np.linalg.norm(np.array([x, y]) - np.array(self.robot_initial_pose[0][:2]))
        z = self.conveyor.get_pose()[0][2]
        pos = [x, y, z]
        return pos, distance

    def sample_linear_motion(self, dist, theta, length):
        """
        return the start (x, y) and end (x, y) of the linear motion

        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 2*pi)
        :param length: the length of the motion
        """
        new_dist = sqrt(dist ** 2 + (length / 2.0) ** 2)
        delta_theta = atan((length / 2.0) / dist)

        theta_1 = theta + delta_theta
        theta_2 = theta - delta_theta

        start_xy = (new_dist * cos(theta_1), new_dist * sin(theta_1))
        end_xy = (new_dist * cos(theta_2), new_dist * sin(theta_2))

        return start_xy, end_xy








class Conveyor:
    def __init__(self, initial_pose, urdf_path, speed):
        self.initial_pose = initial_pose
        self.urdf_path = urdf_path
        self.id = p.loadURDF(self.urdf_path, initial_pose[0], initial_pose[1])
        self.speed = speed

        self.cid = p.createConstraint(parentBodyUniqueId=self.id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                      childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                      parentFramePosition=[0, 0, 0], childFramePosition=initial_pose[0],
                                      childFrameOrientation=initial_pose[1])

        self.start_pose = None
        self.target_pose = None
        self.discretized_trajectory = None
        self.distance = None
        self.wp_target = 0

    def set_pose(self, pose):
        pu.set_pose(self.id, pose)
        self.control_pose(pose)

    def get_pose(self):
        return pu.get_body_pose(self.id)

    def control_pose(self, pose):
        p.changeConstraint(self.cid, jointChildPivot=pose[0], jointChildFrameOrientation=pose[1])

    def reset(self, pose):
        self.set_pose(pose)

    def step(self):
        if self.discretized_trajectory is None or self.wp_target_index == len(self.discretized_trajectory):
            pass
        else:
            self.control_pose(self.discretized_trajectory[self.wp_target_index])
            self.wp_target_index += 1

    def initialize_trajectory(self, target_position):
        self.start_pose = self.get_pose()
        start_position = self.start_pose[0]
        start_orientation = self.start_pose[1]
        self.target_pose = [target_position, start_orientation]

        self.distance = np.linalg.norm(np.array(start_position) - np.array(target_position))
        num_steps = int(self.distance / self.speed * 240)
        position_trajectory = np.linspace(start_position, target_position, num_steps)
        self.discretized_trajectory = [[p, start_orientation] for p in position_trajectory]
        self.wp_target_index = 1



