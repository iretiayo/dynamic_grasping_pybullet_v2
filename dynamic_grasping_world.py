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
import tf_conversions
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from math import pi, cos, sin, sqrt, atan, radians
from ur5_robotiq_pybullet import load_ur_robotiq_robot, UR5RobotiqPybulletController


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

        self.distance_low = 0.15
        self.distance_high = 0.4

        self.grasp_database_path = grasp_database_path
        self.grasps_eef = np.load(os.path.join(self.grasp_database_path, self.target_name, 'grasps_eef.npy'))
        self.graspit_pose_msg = [tf_conversions.toMsg(tf_conversions.fromTf((grasps[:3], grasps[3:]))) for grasps in
                                    self.grasps_eef]
        self.pre_grasps_graspit_pose_msg = [gu.back_off(grasp_pose, self.back_off, approach_dir='x') for grasp_pose in
                                        self.graspit_pose_msg]

        self.graspit_ee_to_moveit_ee_Tf = [[0, 0, 0], [0, 0, 0, 1]]
        self.grasps_eef_pose_msg = [gu.change_end_effector_link(grasps, self.graspit_ee_to_moveit_ee_Tf) for grasps in
                                    self.graspit_pose_msg]
        self.pre_grasps_eef_pose_msg = [gu.back_off(grasp_pose, self.back_off, approach_dir='x') for grasp_pose in
                                        self.grasps_eef_pose_msg]

        # self.reachability_data_dir = reachability_data_dir
        # self.sdf_reachability_space, self.mins, self.step_size, self.dims = gu.get_reachability_space(
        #     self.reachability_data_dir)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF("plane.urdf")
        if 'mico' in self.robot_urdf:
            self.robot = MicoController(self.robot_initial_pose, self.robot_initial_state, self.robot_urdf)
        if 'robotiq' in self.robot_urdf:
            self.robot_id = load_ur_robotiq_robot(self.robot_initial_pose)
            self.robot = UR5RobotiqPybulletController(self.robot_id)
        self.conveyor = Conveyor(self.conveyor_initial_pose, self.conveyor_urdf, self.conveyor_speed)
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])

        p.changeDynamics(self.target, -1, mass=0.5)

        self.reset('initial')  # the reset is needed to simulate for

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
            initial: reset the target to the fixed initial pose, not moving
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
            pu.step(2)
            return target_pose, distance

        elif mode == 'dynamic_linear':
            pu.remove_all_markers()
            self.conveyor.clear_motion()

            distance, theta, length = self.sample_convey_linear_motion()
            self.conveyor.initialize_linear_motion(distance, theta, length)
            conveyor_pose = self.conveyor.start_pose
            target_pose = [[conveyor_pose[0][0], conveyor_pose[0][1], self.target_initial_pose[0][2]],
                           self.sample_target_angle()]
            p.resetBasePositionAndOrientation(self.target, target_pose[0], target_pose[1])
            self.conveyor.set_pose(conveyor_pose)
            self.robot.reset()
            pu.step(2)

            pu.draw_line(self.conveyor.start_pose[0], self.conveyor.target_pose[0])
            return distance, theta, length

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
        grasp_reached = False
        comment = " "

        # planning grasp
        grasp_idx, grasp_planning_time, num_ik_called, pre_grasp, pre_grasp_jv, grasp, grasp_jv = self.plan_grasp(
            predicted_pose, None)
        if grasp_jv is None or pre_grasp_jv is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reached, grasp_planning_time, num_ik_called, "no reachable grasp is found"

        # planning motion
        motion_planning_time, plan = self.plan_arm_motion(pre_grasp_jv)
        if plan is None:
            return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reached, grasp_planning_time, num_ik_called, "no motion found to the planned pre grasp jv"

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
        plan, fraction = self.robot.plan_straight_line(tf_conversions.toMsg(tf_conversions.fromTf(grasp)))
        self.robot.execute_arm_plan(plan, self.realtime)
        grasp_reached = self.robot.equal_conf(self.robot.get_arm_joint_values(), grasp_jv, tol=0.01)  # not valid

        # close and lift
        self.robot.close_gripper(self.realtime)
        lift_pose = tf_conversions.toMsg(tf_conversions.fromTf(grasp))
        lift_pose.position.z += 0.07
        plan, fraction = self.robot.plan_straight_line(lift_pose)
        if fraction != 1.0:
            comment = "lift fraction {} is not 1.0".format(fraction)
        self.robot.execute_arm_plan(plan, self.realtime)
        success = self.check_success()
        pu.remove_all_markers()
        return success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reached, grasp_planning_time, num_ik_called, comment

    def check_success(self):
        if pu.get_body_pose(self.target)[0][2] >= self.target_initial_pose[0][2] + 0.03:
            return True
        else:
            return False

    def dynamic_grasp(self, grasp_threshold, lazy_threshold, close_delay):
        """

        :return attempted_grasp_idx: the executed grasp index
        """
        grasp_idx = None
        done = False
        dynamic_grasp_time = 0
        while not done:
            done = self.check_done()
            target_pose = pu.get_body_pose(self.target)
            predicted_pose = target_pose

            # plan a grasp
            grasp_idx, grasp_planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv \
                = self.plan_grasp(predicted_pose, grasp_idx)
            dynamic_grasp_time += grasp_planning_time
            if planned_grasp_jv is None or planned_pre_grasp_jv is None:
                # print('no reachable grasp found')
                self.step(grasp_planning_time, None, None)
                continue
            self.step(grasp_planning_time, None, None)
            pu.create_arrow_marker(planned_pre_grasp, color_index=grasp_idx)

            # plan a motion
            distance = np.linalg.norm(np.array(self.robot.get_eef_pose()[0]) - np.array(planned_pre_grasp[0]))
            if distance > lazy_threshold and self.robot.arm_discretized_plan is not None:
                # print("lazy plan")
                continue
            motion_planning_time, plan = self.plan_arm_motion(planned_pre_grasp_jv)
            dynamic_grasp_time += motion_planning_time
            if plan is None:
                # print('no motion is found')
                self.step(motion_planning_time, None, None)
                continue
            self.step(motion_planning_time, plan, None)

            # check can grasp or not
            if self.robot.equal_conf(self.robot.get_arm_joint_values(), planned_pre_grasp_jv, tol=grasp_threshold):
                motion_planning_time, arm_motion_plan, gripper_motion_plan = self.plan_approach_motion(planned_grasp_jv)
                dynamic_grasp_time += motion_planning_time
                self.execute_appraoch_and_grasp(arm_motion_plan, gripper_motion_plan, close_delay)
                self.execute_lift()
                return self.check_success(), grasp_idx, dynamic_grasp_time
        return False, None, dynamic_grasp_time

    def plan_approach_motion(self, grasp_jv):
        """ Plan the discretized approach motion for both arm and gripper """
        predicted_period = 0.2
        start_time = time.time()

        if self.robot.arm_discretized_plan is not None:
            future_target_index = min(int(predicted_period * 240 + self.robot.arm_wp_target_index),
                                      len(self.robot.arm_discretized_plan) - 1)
            start_joint_values = self.robot.arm_discretized_plan[future_target_index]
            arm_discretized_plan = self.robot.plan_arm_joint_values_simple(grasp_jv,
                                                                           start_joint_values=start_joint_values)
        else:
            arm_discretized_plan = self.robot.plan_arm_joint_values_simple(grasp_jv)

        # there is no gripper discretized plan
        gripper_discretized_plan = self.robot.plan_gripper_joint_values(self.robot.CLOSED_POSITION)

        planning_time = time.time() - start_time
        # print("Planning a motion takes {:.6f}".format(planning_time))
        return planning_time, arm_discretized_plan, gripper_discretized_plan

    def execute_appraoch_and_grasp(self, arm_plan, gripper_plan, delay):
        num_delay_steps = int(delay * 240.0)
        arm_len = len(arm_plan)
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
                    # print("Planning a grasp takes {:.6f}".format(planning_time))
                    return old_grasp_idx, planning_time, num_ik_called, planned_pre_grasp, planned_pre_grasp_jv, planned_grasp, planned_grasp_jv

        if self.disable_reachability:
            grasp_order_idxs = np.random.permutation(np.arange(len(self.grasps_eef_pose_msg)))
        else:
            pre_grasps_graspit_poses_in_world = [
                tf_conversions.toTf(tf_conversions.fromTf(target_pose) * tf_conversions.fromMsg(g)) for g in
                self.pre_grasps_graspit_pose_msg]
            sdf_values = gu.get_reachability_of_grasps_pose_2d(pre_grasps_graspit_poses_in_world,
                                                               self.sdf_reachability_space,
                                                               self.mins,
                                                               self.step_size,
                                                               self.dims)
            grasp_order_idxs = np.argsort(sdf_values)[::-1]
        for num_ik_called, grasp_idx in enumerate(grasp_order_idxs):
            if num_ik_called == self.max_check:
                break

            planned_pre_grasp_in_object = self.pre_grasps_eef_pose_msg[grasp_idx]
            planned_pre_grasp = tf_conversions.toTf(tf_conversions.fromTf(target_pose) * tf_conversions.fromMsg(planned_pre_grasp_in_object))
            planned_pre_grasp_jv = self.robot.get_arm_ik(planned_pre_grasp)
            if planned_pre_grasp_jv is None:
                continue

            planned_grasp_in_object = self.grasps_eef_pose_msg[grasp_idx]
            planned_grasp = tf_conversions.toTf(tf_conversions.fromTf(target_pose) * tf_conversions.fromMsg(planned_grasp_in_object))
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
        # print("Planning a grasp takes {:.6f}".format(planning_time))
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

        # print("Planning a motion takes {:.6f}".format(planning_time))
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

    @staticmethod
    def sample_target_angle():
        """ return quaternion """
        angle = np.random.uniform(-pi, pi)
        orientation = p.getQuaternionFromEuler([0, 0, angle])
        return orientation

    def sample_convey_linear_motion(self, dist=None, theta=None, length=None):
        if dist is None:
            dist = np.random.uniform(low=self.distance_low, high=self.distance_high)
        if theta is None:
            theta = np.random.uniform(low=0, high=360)
        if length is None:
            length = 1.0
        return dist, theta, length

    def check_done(self):
        done = False
        if self.conveyor.wp_target_index == len(self.conveyor.discretized_trajectory):
            # conveyor trajectory finishes
            done = True
        if pu.get_body_pose(self.target)[0][2] < self.target_initial_pose[0][2] - 0.03:
            # target knocked down
            done = True
        return done


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
        self.wp_target_index = 0
        self.distance = None
        self.theta = None
        self.length = None

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

    def initialize_linear_motion(self, dist, theta, length):
        """
        :param dist: distance to robot center,
        :param theta: the angle of rotation, (0, 360)
        :param length: the length of the motion
        """
        self.distance = dist
        self.theta = theta
        self.length = length
        # uses the z value and orientation of the current pose
        z = self.get_pose()[0][-1]
        orientation = self.get_pose()[1]
        # compute start xy and end xy
        new_dist = sqrt(dist ** 2 + (length / 2.0) ** 2)
        delta_theta = atan((length / 2.0) / dist)

        theta_1 = radians(self.theta) + delta_theta
        theta_2 = radians(self.theta) - delta_theta

        start_xy = [new_dist * cos(theta_1), new_dist * sin(theta_1)]
        target_xy = [new_dist * cos(theta_2), new_dist * sin(theta_2)]
        start_position = start_xy + [z]
        target_position = target_xy + [z]

        self.start_pose = [start_position, orientation]
        self.target_pose = [target_position, orientation]

        num_steps = int(self.length / self.speed * 240)
        position_trajectory = np.linspace(start_position, target_position, num_steps)
        self.discretized_trajectory = [[p, orientation] for p in position_trajectory]

    def clear_motion(self):
        self.start_pose = None
        self.target_pose = None
        self.discretized_trajectory = None
        self.wp_target_index = 0
        self.distance = None
        self.theta = None
        self.length = None
