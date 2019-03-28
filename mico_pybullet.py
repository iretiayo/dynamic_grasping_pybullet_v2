import pybullet as p
from collections import namedtuple
from mico_moveit import MicoMoveit
import time
import numpy as np
import rospy
import tf_conversions

import actionlib
import pybullet_trajectory_execution.msg


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

        self.client = actionlib.SimpleActionClient('trajectory_execution',
                                                   pybullet_trajectory_execution.msg.TrajectoryAction)
        self.client.wait_for_server()
        self.goal_id = 0
        self.seq = None
        self.start_time_stamp = None

    ### Control
    def move_arm_eef_pose(self, pose, plan=True):
        """ Plan and then execute, using goal eff pose """
        pose_joint_values = self.get_arm_ik(pose, avoid_collisions=plan)
        self.move_arm_joint_values(pose_joint_values, plan)

    def move_arm_joint_values(self, goal_joint_values, plan=True):
        """
        Plan and then execute, using goal joint values.
        :param goal_joint_values: goal arm joint values
        :param plan: if True, use moveit as backend to plan a collision free path; otherwise
            just linear interpolation in configuration space. plan = False should only be used for short
            motion where we are confident that no cillision is on the way.
        """
        if plan:
            position_trajectory, motion_plan = self.plan_arm_joint_values(goal_joint_values)
            if position_trajectory is not None:
                self.execute_arm_trajectory(position_trajectory, motion_plan)
            else:
                print("No path found!")
        else:
            num_steps = 10  # number of waypoints
            start_joint_values = self.get_arm_joint_values()
            diff = np.array(goal_joint_values) - np.array(start_joint_values)
            diff[diff > 2*np.pi] -= 2*np.pi
            diff[diff < -2*np.pi] += 2*np.pi
            converted_goal_joint_values = np.array(start_joint_values) + diff
            print('\n\n\nUsing custom move to pick')
            print(diff)
            position_trajectory = np.linspace(start_joint_values, converted_goal_joint_values, num_steps)
            self.execute_arm_trajectory(position_trajectory, None)

    def reset_arm_joint_values(self, joint_values):
        arm_idx = self.GROUP_INDEX['arm']
        for i, v in zip(arm_idx, joint_values):
            p.resetJointState(self.id, i, v)

    def reset_arm_eef_pose(self, pose):
        pass

    def move_gripper_joint_values(self, joint_values, duration=1.0, num_steps=10):
        """ this method has nothing to do with moveit """
        start_joint_values = self.get_gripper_joint_values()
        goal_joint_values = joint_values
        position_trajectory = np.linspace(start_joint_values, goal_joint_values, num_steps)
        for i in range(num_steps):
            p.setJointMotorControlArray(self.id, self.GROUP_INDEX['gripper'], p.POSITION_CONTROL,
                                        position_trajectory[i], forces=[200] * 2)
            time.sleep(float(duration) / float(num_steps))

    def open_gripper(self):
        self.move_gripper_joint_values([0.0, 0.0])

    def close_gripper(self):
        # self.move_gripper_joint_values([2,0, 2.0]) # over close
        # self.move_gripper_joint_values([1.5, 1.5]) # just close
        self.move_gripper_joint_values([1.3, 1.3])  # not close

    def cartesian_control(self, x=0.0, y=0.0, z=0.0, frame="world"):
        """
        Only for small motion
        :param frame: "eef" or "world"
        """
        if frame == "eef":
            pose_2d = self.get_arm_eef_pose()
            world_T_old = tf_conversions.toMatrix(tf_conversions.fromTf(pose_2d))
            old_T_new = tf_conversions.toMatrix(tf_conversions.fromTf(((x, y, z), (0, 0, 0, 1))))
            world_T_new = world_T_old.dot(old_T_new)
            pose_2d_new = tf_conversions.toTf(tf_conversions.fromMatrix(world_T_new))
            self.move_arm_eef_pose(pose_2d_new, plan=False)
        elif frame == "world":
            pose_2d_new = self.get_arm_eef_pose()
            pose_2d_new[0][0] += x
            pose_2d_new[0][1] += y
            pose_2d_new[0][2] += z
            self.move_arm_eef_pose(pose_2d_new, plan=False)
        else:
            raise TypeError("not supported frame: {}".format(frame))

    def grasp(self, pre_g_pose, dynamic):
        if not dynamic:
            rospy.loginfo("start grasping")
            g_pose = self.back_off(pre_g_pose, -0.05)
            self.move_arm_eef_pose(g_pose, plan=False)  # sometimes this motion is werid? rarely
            time.sleep(1)  # give sometime to move before closing
            self.close_gripper()
            self.cartesian_control(z=0.05)
            time.sleep(1)
            ## simply attach an object will not enable collision checking with it
            # touch_links = mc.mico_moveit.robot.get_link_names('gripper')
            # eef_link = mc.mico_moveit.arm_commander_group.get_end_effector_link()
            # mc.mico_moveit.scene.attach_box(eef_link, 'cube', touch_links=touch_links)
            # self.move_arm_joint_values(self.HOME)
        else:
            pass

    @staticmethod
    def back_off(pose_2d, offset):
        """
        Back off a grasp pose in world; namely, calculate the new pose after moving the eef along its z axis.

        :param pose_2d: world pose, [[x, y, z], [x, y, z, w]]
        :param offset: the amount of distance to back off
        """
        world_T_old = tf_conversions.toMatrix(tf_conversions.fromTf(pose_2d))
        old_T_new = tf_conversions.toMatrix(tf_conversions.fromTf(((0, 0, -offset), (0, 0, 0, 1))))
        world_T_new = world_T_old.dot(old_T_new)
        pose_2d_new = tf_conversions.toTf(tf_conversions.fromMatrix(world_T_new))
        return pose_2d_new

    @staticmethod
    def extract_plan(plan):
        """ Extract numpy arrays of position, velocity and time trajectories from moveit plan """
        points = plan.joint_trajectory.points
        position_trajectory = []
        velocity_trajectory = []
        time_trajectory = []
        for p in points:
            position_trajectory.append(list(p.positions))
            velocity_trajectory.append(list(p.velocities))
            time_trajectory.append(p.time_from_start.to_sec())
        return np.array(position_trajectory), np.array(velocity_trajectory), np.array(time_trajectory)

    # abandoned
    @staticmethod
    def convert_plan(plan, start_joint_values):
        """ Convert plan returned by moveit to work with current start joint values """
        position_trajectory, velocity_trajectory, time_trajectory = MicoController.extract_plan(plan)

        # convert position trajectory to work with current joint values
        new_position_trajectory = MicoController.convert_position_trajectory(position_trajectory, start_joint_values)

        return new_position_trajectory, velocity_trajectory, time_trajectory

    # abandoned
    @staticmethod
    def convert_position_trajectory(position_trajectory, start_joint_values):
        """
        Convert a position trajectory to work with current joint values
        :param position_trajectory: 2d np array
        :param start_joint_values: a list of values, current arm joint values
        """
        # convert position trajectory to work with current joint values
        diff = np.array(start_joint_values) - position_trajectory[0]
        new_position_trajectory = position_trajectory + diff
        return new_position_trajectory

    @staticmethod
    def process_plan(plan, start_joint_values):
        # convert position trajectory to work with current joint values
        diff = np.array(start_joint_values) - np.array(plan.joint_trajectory.points[0].positions)
        for p in plan.joint_trajectory.points:
            p.positions = (np.array(p.positions) + diff).tolist()
        position_trajectory, velocity_trajectory, time_trajectory = MicoController.extract_plan(plan)
        return position_trajectory, plan

    @staticmethod
    def interpolate_plan_at_time(plan, time_point):
        position_trajectory, velocity_trajectory, time_trajectory = MicoController.extract_plan(plan)
        idx = min(np.argmin(np.abs(time_trajectory - time_point)), len(time_trajectory) - 1)
        if time_point < time_trajectory[idx]:
            idx -= 1
        if idx == len(time_trajectory) - 1:
            return position_trajectory[idx]

        diff = position_trajectory[idx + 1] - position_trajectory[idx]
        time_fraction = (time_point - time_trajectory[idx]) / (time_trajectory[idx + 1] - time_trajectory[idx])
        return position_trajectory[idx] + time_fraction * diff

    def plan_arm_joint_values(self, goal_joint_values, start_joint_values=None):
        """
        Plan a trajectory from current joint values to goal joint values
        :param goal_joint_values: a list of goal joint values
        :param start_joint_values: a list of start joint values; if None, use current values
        """
        if start_joint_values is None:
            start_joint_values = self.get_arm_joint_values()

        start_joint_values_converted = self.mico_moveit.convert_range(start_joint_values)
        goal_joint_values_converted = self.mico_moveit.convert_range(goal_joint_values)

        plan = self.mico_moveit.plan(start_joint_values_converted, goal_joint_values_converted) # STOMP does not convert goal joint values
        # check if there exists a plan
        if len(plan.joint_trajectory.points) == 0:
            return None, None

        position_trajectory, plan = MicoController.process_plan(plan, start_joint_values)
        return position_trajectory, plan

    def plan_arm_eef_pose_old(self, pose):
        pose_joint_values = self.get_arm_ik(pose)
        return self.plan_arm_joint_values(pose_joint_values)

    def plan_arm_eef_pose(self, ee_pose, start_joint_values=None):
        if start_joint_values is None:
            start_joint_values = self.get_arm_joint_values()

        plan = self.mico_moveit.plan_ee_pose(start_joint_values, ee_pose)
        # check if there exists a plan
        if len(plan.joint_trajectory.points) == 0:
            return None, None

        position_trajectory, plan = MicoController.process_plan(plan, start_joint_values)
        return position_trajectory, plan

    def execute_arm_trajectory(self, position_trajectory, motion_plan):
        goal = pybullet_trajectory_execution.msg.TrajectoryGoal(
            waypoints=[pybullet_trajectory_execution.msg.Waypoint(waypoint) for waypoint in position_trajectory],
            robot_id=self.id,
            joint_indices=self.GROUP_INDEX['arm'],
            goal_id=self.goal_id,
            motion_plan=motion_plan)
        self.client.send_goal(goal, feedback_cb=self.feedback_cb)
        self.goal_id += 1

    def feedback_cb(self, feedback):
        # rospy.loginfo("receive feedback: " + str(feedback))
        self.seq = feedback.seq
        self.start_time_stamp = feedback.start_time_stamp

    ''' Helper functions '''

    def get_arm_eef_pose(self):
        """
        :return: pose_2d, [[x, y, z], [x, y, x, w]]
        """
        link_state = self.get_link_state(self.ARM_EEF_INDEX)
        position = list(link_state.linkWorldPosition)
        orn = list(link_state.linkWorldOrientation)
        return [position, orn]

    def get_arm_ik(self, pose_2d, timeout=0.01, avoid_collisions=True):
        gripper_joint_values = self.get_gripper_joint_values()
        arm_joint_values = self.get_arm_joint_values()
        j = self.mico_moveit.get_arm_ik(pose_2d, timeout, avoid_collisions, arm_joint_values, gripper_joint_values)
        if j is None:
            # print("No ik exists!")
            return None
        else:
            # return MicoMoveit.convert_range(j)
            return j

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
        return [self.get_joint_info(i).jointName for i in range(self.num_joints) if
                self.get_joint_info(i).jointType != p.JOINT_FIXED]

    def get_joint_index(self, name):
        return self.get_joint_names().index(name)
