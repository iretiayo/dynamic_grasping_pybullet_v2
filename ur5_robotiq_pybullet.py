from collections import namedtuple
import pybullet as p

import rospy
import rospkg
import moveit_commander as mc
from moveit_msgs.srv import GetPositionIK, GetPositionFK

from ur5_robotiq_moveit import UR5RobotiqMoveIt

import numpy as np
import os
import time

import pybullet_utils as pu

def load_ur_robotiq_robot(robot_initial_pose):
    # load robot
    urdf_dir = os.path.join(rospkg.RosPack().get_path('ur5_robotiq_description'), 'urdf')
    urdf_filepath = os.path.join(urdf_dir, 'ur5_robotiq.urdf')
    xacro_filepath = os.path.join(urdf_dir, 'ur5_robotiq.xacro')
    if not os.path.exists(urdf_filepath):
        cmd = 'rosrun xacro xacro --inorder {} -o {}'.format(xacro_filepath, urdf_filepath)
        os.system(cmd)
        robotiq_description_dir = rospkg.RosPack().get_path('robotiq_2f_85_gripper_visualization')
        sed_cmd = "sed -i 's|{}|{}|g' {}".format('package://robotiq_2f_85_gripper_visualization',
                                                 robotiq_description_dir, urdf_filepath)
        os.system(sed_cmd)
        ur5_description_dir = rospkg.RosPack().get_path('ur_description')
        sed_cmd = "sed -i 's|{}|{}|g' {}".format('package://ur_description', ur5_description_dir, urdf_filepath)
        os.system(sed_cmd)

    robot_id = p.loadURDF(urdf_filepath, basePosition=robot_initial_pose[0], baseOrientation=robot_initial_pose[1],
                          flags=p.URDF_USE_SELF_COLLISION)
    return robot_id

Motion = namedtuple('Motion', ['position_trajectory', 'time_trajectory', 'velocity_trajectory'])

class UR5RobotiqPybulletController(object):
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
        'arm': ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'],
        'gripper': ['finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint', 'right_outer_knuckle_joint',
                    'right_inner_knuckle_joint', 'right_inner_finger_joint']
    }
    HOME = [0, -0.8227210029571718, -0.130, -0.660, 0, 1.62]
    OPEN_POSITION = [0] * 6
    CLOSED_POSITION = 0.72 * np.array([1, 1, -1, 1, 1, -1])

    JOINT_INDICES_DICT = {}
    EE_LINK_NAME = 'ee_link'

    TIP_LINK = "ee_link"
    BASE_LINK = "base_link"
    ARM = "manipulator"
    GRIPPER = "gripper"
    ARM_JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    GRIPPER_JOINT_NAMES = ['finger_joint']

    # this is read from moveit_configs joint_limits.yaml
    MOVEIT_ARM_MAX_VELOCITY = [3.15, 3.15, 3.15, 3.15, 3.15, 3.15]

    def __init__(self, robot_id):
        self.id = robot_id
        self.initial_joint_values = self.HOME
        self.num_joints = p.getNumJoints(self.id)

        joint_infos = [p.getJointInfo(robot_id, joint_index) for joint_index in range(p.getNumJoints(robot_id))]
        self.JOINT_INDICES_DICT = {entry[1]: entry[0] for entry in joint_infos}
        self.GROUP_INDEX = {key: [self.JOINT_INDICES_DICT[joint_name] for joint_name in self.GROUPS[key]] for key in
                            self.GROUPS}
        self.EEF_LINK_INDEX = pu.link_from_name(robot_id, self.EE_LINK_NAME)

        self.moveit = UR5RobotiqMoveIt()

        self.arm_commander_group = mc.MoveGroupCommander(self.ARM)
        self.robot = mc.RobotCommander()
        self.scene = mc.PlanningSceneInterface()
        rospy.sleep(2)

        self.arm_difference_fn = pu.get_difference_fn(self.id, self.GROUP_INDEX['arm'])
        self.reset()

    def reset(self):
        self.set_arm_joints(self.initial_joint_values)
        self.set_gripper_joints(self.OPEN_POSITION)
        self.clear_scene()
        self.arm_discretized_plan = None
        self.gripper_discretized_plan = None
        self.arm_wp_target_index = 0
        self.gripper_wp_target_index = 0

    def set_arm_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['arm'], joint_values)
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def control_arm_joints(self, joint_values):
        pu.control_joints(self.id, self.GROUP_INDEX['arm'], joint_values)

    def set_gripper_joints(self, joint_values):
        pu.set_joint_positions(self.id, self.GROUP_INDEX['gripper'], joint_values)
        pu.control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values)

    def control_gripper_joints(self, joint_values):
        pu.control_joints(self.id, self.GROUP_INDEX['gripper'], joint_values)

    def close_gripper(self, realtime=False):
        waypoints = self.plan_gripper_joint_values(self.CLOSED_POSITION)
        self.execute_gripper_plan(waypoints, realtime)

    def plan_gripper_joint_values(self, goal_joint_values, start_joint_values=None):
        if start_joint_values is None:
            start_joint_values = self.get_gripper_joint_values()
        num_steps = 240
        discretized_plan = np.linspace(start_joint_values, goal_joint_values, num_steps)
        return discretized_plan

    def clear_scene(self):
        self.moveit.clear_scene()

    def get_arm_ik(self, pose_2d, timeout=0.1, avoid_collisions=True, arm_joint_values=None, gripper_joint_values=None):

        start_joint_values = self.get_arm_joint_values() if arm_joint_values is None else arm_joint_values
        gripper_joint_values = self.get_gripper_joint_values() if gripper_joint_values is None else gripper_joint_values

        return self.moveit.get_arm_ik(pose_2d, timeout, avoid_collisions, start_joint_values, gripper_joint_values)

    def plan_arm_joint_values(self, goal_joint_values, start_joint_values=None, maximum_planning_time=0.5):
        if start_joint_values is None:
            start_joint_values = self.get_arm_joint_values()

        start_joint_values_converted = UR5RobotiqPybulletController.convert_range(start_joint_values)
        goal_joint_values_converted = UR5RobotiqPybulletController.convert_range(goal_joint_values)

        # STOMP does not convert goal joint values
        moveit_plan = self.moveit.plan(start_joint_values_converted, goal_joint_values_converted,
                                       maximum_planning_time=maximum_planning_time)
        # check if there exists a plan
        if len(moveit_plan.joint_trajectory.points) == 0:
            return None

        motion_plan = UR5RobotiqPybulletController.process_plan(moveit_plan, start_joint_values)
        discretized_plan = UR5RobotiqPybulletController.discretize_plan(motion_plan)
        return discretized_plan

    def plan_arm_joint_values_simple(self, goal_joint_values, start_joint_values=None):
        """ Linear interpolation between joint_values """
        start_joint_values = self.get_arm_joint_values() if start_joint_values is None else start_joint_values

        diffs = self.arm_difference_fn(goal_joint_values, start_joint_values)
        steps = np.abs(np.divide(diffs, self.MOVEIT_ARM_MAX_VELOCITY)) * 240
        num_steps = int(max(steps))

        waypoints = [start_joint_values]
        for i in range(num_steps):
            waypoints.append(list(((float(i) + 1.0) / float(num_steps)) * np.array(diffs) + start_joint_values))
        print(self.adapt_conf(goal_joint_values, waypoints[-1]))
        return waypoints

    def plan_straight_line(self, eef_pose):
        start_joint_values = self.get_arm_joint_values()
        start_joint_values_converted = self.convert_range(start_joint_values)

        # TODO: avoid_collisions should be allow touch object
        moveit_plan, fraction = self.moveit.plan_straight_line(start_joint_values_converted, eef_pose,
                                                               avoid_collisions=False)

        # check if there exists a plan
        if len(moveit_plan.joint_trajectory.points) == 0:
            return None, fraction
        plan = self.process_plan(moveit_plan, start_joint_values)
        discretized_plan = UR5RobotiqPybulletController.discretize_plan(plan)
        return discretized_plan, fraction

    @staticmethod
    def convert_range(joint_values):
        """ Convert continuous joint to range [-pi, pi] """
        circular_idx = [0, 3, 4, 5]
        new_joint_values = []
        for i, v in enumerate(joint_values):
            if i in circular_idx:
                new_joint_values.append(pu.wrap_angle(v))
            else:
                new_joint_values.append(v)
        return new_joint_values

    @staticmethod
    def process_plan(moveit_plan, start_joint_values):
        """
        convert position trajectory to work with current joint values
        :param moveit_plan: MoveIt plan
        :return plan: Motion
        """
        diff = np.array(start_joint_values) - np.array(moveit_plan.joint_trajectory.points[0].positions)
        for p in moveit_plan.joint_trajectory.points:
            p.positions = (np.array(p.positions) + diff).tolist()
        plan = UR5RobotiqPybulletController.extract_plan(moveit_plan)
        return plan

    @staticmethod
    def discretize_plan(motion_plan):
        """ return np array """
        discretized_plan = np.zeros((0, 6))
        for i in range(len(motion_plan.position_trajectory) - 1):
            num_steps = (motion_plan.time_trajectory[i + 1] - motion_plan.time_trajectory[i]) * 240
            segment = np.linspace(motion_plan.position_trajectory[i], motion_plan.position_trajectory[i + 1], num_steps)
            if i + 1 == len(motion_plan.position_trajectory) - 1:
                discretized_plan = np.vstack((discretized_plan, segment))
            else:
                discretized_plan = np.vstack((discretized_plan, segment[:-1]))
        return discretized_plan

    @staticmethod
    def process_discretized_plan(discretized_plan, start_joint_values):
        """
        convert discretized plan to work with current joint values
        :param discretized_plan: discretized plan, list of waypoints
        :return plan: Motion
        """
        diff = np.array(start_joint_values) - np.array(discretized_plan[0])
        new_discretized_plan = []
        for wp in discretized_plan:
            new_discretized_plan.append((np.array(wp) + diff).tolist())
        return new_discretized_plan

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


    # execution
    def execute_arm_plan(self, plan, realtime=False):
        """
        execute a discretized arm plan (list of waypoints)
        """
        for wp in plan:
            self.control_arm_joints(wp)
            p.stepSimulation()
            if realtime:
                time.sleep(1. / 240.)
        pu.step(2)

    def execute_gripper_plan(self, plan, realtime=False):
        """
        execute a discretized gripper plan (list of waypoints)
        """
        for wp in plan:
            self.control_gripper_joints(wp)
            p.stepSimulation()
            if realtime:
                time.sleep(1. / 240.)
        pu.step(2)

    def equal_conf(self, conf1, conf2, tol=0):
        adapted_conf2 = self.adapt_conf(conf2, conf1)
        return np.allclose(conf1, adapted_conf2, atol=tol)

    def adapt_conf(self, conf2, conf1):
        """ adapt configuration 2 to configuration 1"""
        diff = self.arm_difference_fn(conf2, conf1)
        adapted_conf2 = np.array(conf1) + np.array(diff)
        return adapted_conf2.tolist()



    def reset_joint_values(self, joint_indices, joint_values):
        for i, v in zip(joint_indices, joint_values):
            p.resetJointState(self.id, i, v)

    def reset_arm_joint_values(self, joint_values):
        self.reset_joint_values(self.GROUP_INDEX['arm'], joint_values)

    def reset_gripper_joint_values(self, joint_values):
        self.reset_joint_values(self.GROUP_INDEX['gripper'], joint_values)

    def set_group_joint_values(self, group_joint_indices, joint_values):
        p.setJointMotorControlArray(self.id, group_joint_indices, p.POSITION_CONTROL, joint_values,
                                    forces=[500] * len(joint_values))

    def set_arm_joint_values(self, joint_values):
        self.set_group_joint_values(self.GROUP_INDEX['arm'], joint_values)

    def set_gripper_joint_values(self, joint_values=(0,)*6):
        self.set_group_joint_values(self.GROUP_INDEX['gripper'], joint_values)

    def get_joint_state(self, joint_index):
        return self.JointState(*p.getJointState(self.id, joint_index))

    def get_arm_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['arm']]

    def get_gripper_joint_values(self):
        return [self.get_joint_state(i).jointPosition for i in self.GROUP_INDEX['gripper']]

    def execute_arm_motion_plan(self, motion_plan):
        try:
            rospy.loginfo("length of trajectory is: {}".format(len(motion_plan.joint_trajectory.points)))

            jt_points = motion_plan.joint_trajectory.points
            frequency = 50.
            rate = rospy.Rate(frequency)
            start_time = rospy.Time.now()
            next_point_idx = 1  # TODO what if trajectory has only 1 point?
            rospy.loginfo('moving to next trajectory point {}'.format(next_point_idx))
            while True:
                time_since_start = rospy.Time.now() - start_time
                time_diff = (jt_points[next_point_idx].time_from_start - time_since_start).to_sec()

                # handle redundant first points
                if time_diff == 0:
                    rospy.loginfo("ignore trajectory point {}".format(next_point_idx))
                    next_point_idx += 1
                    rospy.loginfo('moving to next trajectory point {}'.format(next_point_idx))
                    continue

                num_steps = max((time_diff * frequency), 1)

                current_jv = self.get_arm_joint_values()
                next_jv = current_jv + (np.array(jt_points[next_point_idx].positions) - current_jv) / num_steps
                self.set_arm_joint_values(next_jv)
                if (rospy.Time.now() - start_time).to_sec() > jt_points[next_point_idx].time_from_start.to_sec():
                    next_point_idx += 1
                    rospy.loginfo('moving to next trajectory point {}'.format(next_point_idx))
                    # import ipdb; ipdb.set_trace()
                if next_point_idx == len(jt_points):
                    break
                rate.sleep()
            rospy.loginfo('Trajectory took {} secs instead of {} secs'.format((rospy.Time.now() - start_time).to_sec(),
                                                                              jt_points[-1].time_from_start.to_sec()))
        except Exception as e:
            print("exception in execute_motion_plan catched")
            print(e)

    def move_gripper_joint_values(self, joint_values, duration=1.0, num_steps=10):
        """ this method has nothing to do with moveit """
        start_joint_values = self.get_gripper_joint_values()
        goal_joint_values = joint_values
        position_trajectory = np.linspace(start_joint_values, goal_joint_values, num_steps)
        for i in range(num_steps):
            p.setJointMotorControlArray(self.id, self.GROUP_INDEX['gripper'], p.POSITION_CONTROL,
                                        position_trajectory[i], forces=[200] * len(joint_values))
            rospy.sleep(duration / num_steps)
