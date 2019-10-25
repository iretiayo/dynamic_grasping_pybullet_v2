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
from collections import namedtuple

Motion = namedtuple('Motion', ['position_trajectory', 'time_trajectory', 'velocity_trajectory'])


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

        # the service names have to be this
        self.arm_ik_svr = rospy.ServiceProxy('compute_ik', GetPositionIK)
        self.arm_fk_svr = rospy.ServiceProxy('compute_fk', GetPositionFK)

        # for stepping the robot
        self.motion_plan = None

    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.robot_id, self.GROUP_INDEX['arm'], joint_values)
        pu.control_joints(self.robot_id, self.GROUP_INDEX['arm'], joint_values)

    def control_arm_joints(self, joint_values):
        pu.control_joints(self.robot_id, self.GROUP_INDEX['arm'], joint_values)

    def compute_next_action(self, object_pose, ):
        pass

    def step(self):
        """ step the robot for 1/240 second """
        # calculate the latest conf and control array
        if self.motion_plan is None:
            pass
        else:
            pass

    def update_motion_plan(self, motion_plan):
        self.motion_plan = motion_plan
        ## TODO make a step motion plan for each 1/240 time step
        self.target_waypoint = self.motion_plan[0]

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

    @staticmethod
    def process_plan(moveit_plan, start_joint_values):
        """
        convert position trajectory to work with current joint values
        :param plan: MoveIt plan
        :return plan: Motion
        """
        diff = np.array(start_joint_values) - np.array(moveit_plan.joint_trajectory.points[0].positions)
        for p in moveit_plan.joint_trajectory.points:
            p.positions = (np.array(p.positions) + diff).tolist()
        plan = MicoController.extract_plan(moveit_plan)
        return plan

    @staticmethod
    def extract_plan(moveit_plan):
        """
        Extract numpy arrays of position, velocity and time trajectories from moveit plan,
        and return Motion object
        """
        points = moveit_plan.joint_trajectory.points
        position_trajectory = []
        velocity_trajectory = []
        time_trajectory = []
        for p in points:
            position_trajectory.append(list(p.positions))
            velocity_trajectory.append(list(p.velocities))
            time_trajectory.append(p.time_from_start.to_sec())
        return Motion(np.array(position_trajectory), np.array(time_trajectory), np.array(velocity_trajectory))

    def close_gripper(self):
        num_steps = 240
        waypoints = np.linspace(self.OPEN_POSITION, self.CLOSED_POSITION, num_steps)
        for wp in waypoints:
            pu.control_joints(self.robot_id, self.GROUP_INDEX['gripper'], wp)
            p.stepSimulation()
        pu.step()

    def plan_arm_joint_values(self, goal_joint_values, start_joint_values=None, maximum_planning_time=0.5):
        """
        Plan a trajectory from current joint values to goal joint values
        :param goal_joint_values: a list of goal joint values
        :param start_joint_values: a list of start joint values; if None, use current values
        :return plan: Motion
        """
        if start_joint_values is None:
            start_joint_values = self.get_arm_joint_values()

        start_joint_values_converted = self.convert_range(start_joint_values)
        goal_joint_values_converted = self.convert_range(goal_joint_values)

        moveit_plan = self.plan_arm_joint_values_ros(start_joint_values_converted, goal_joint_values_converted,
                                                     maximum_planning_time=maximum_planning_time)  # STOMP does not convert goal joint values
        # check if there exists a plan
        if len(moveit_plan.joint_trajectory.points) == 0:
            return None, None

        plan = MicoController.process_plan(moveit_plan, start_joint_values)
        return plan

    def plan_arm_joint_values_ros(self, start_joint_values, goal_joint_values, maximum_planning_time=0.5):
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

    def plan_straight_line(self, goal_eef_pose, start_joint_values=None, ee_step=0.05, jump_threshold=3.0,
                           avoid_collisions=True):
        if start_joint_values is None:
            start_joint_values = self.get_arm_joint_values()

        # moveit will do the conversion internally
        plan, fraction = self.mico_moveit.plan_straight_line(start_joint_values, goal_eef_pose, ee_step=ee_step,
                                                             jump_threshold=jump_threshold,
                                                             avoid_collisions=avoid_collisions)

        # print("plan length: {}, fraction: {}".format(len(plan.joint_trajectory.points), fraction))

        # check if there exists a plan
        if len(plan.joint_trajectory.points) == 0:
            return None, None, fraction

        position_trajectory, plan = MicoController.process_plan(plan, start_joint_values)
        return position_trajectory, plan, fraction

    def plan_straight_line_ros(self, start_joint_values, end_eef_pose, ee_step=0.05, jump_threshold=3.0,
                               avoid_collisions=True):
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
        # TODO plan should take in gripper joint values for start state
        # TODO reduce step

        # from scratch
        # joint_state = JointState()
        # joint_state.name = self.ARM_JOINT_NAMES
        # joint_state.position = start_joint_values
        # moveit_robot_state = RobotState()
        # moveit_robot_state.joint_state = joint_state

        # using current state, including all other joint info
        start_robot_state = self.robot.get_current_state()
        start_robot_state.joint_state.name = self.ARM_JOINT_NAMES
        start_robot_state.joint_state.position = start_joint_values

        self.arm_commander_group.set_start_state(start_robot_state)

        start_eef_pose = self.get_arm_fk(start_joint_values)
        plan, fraction = self.arm_commander_group.compute_cartesian_path(
            [start_eef_pose, end_eef_pose],
            ee_step,
            jump_threshold,
            avoid_collisions)
        # remove the first redundant point
        plan.joint_trajectory.points = plan.joint_trajectory.points[1:]
        # speed up the trajectory
        for p in plan.joint_trajectory.points:
            p.time_from_start = rospy.Duration.from_sec(p.time_from_start.to_sec() / 1.5)
        return plan, fraction

    def violate_limits(self, joint_values):
        return pu.violates_limits(self.robot_id, self.GROUPS['arm'], joint_values)
