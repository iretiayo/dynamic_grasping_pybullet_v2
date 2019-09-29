from collections import namedtuple
import pybullet as p

import rospy
import numpy as np


class MicoControllerSimple(object):
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
    HOME = [4.80469, 2.92482, 1.002, 4.20319, 1.4458, 1.3233]
    OPEN_GRIPPER = [0, 0]
    CLOSE_GRIPPER = [1.3, 1.3]

    def __init__(self, mico_id):
        self.id = mico_id
        self.num_joints = p.getNumJoints(self.id)

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

    def set_gripper_joint_values(self, joint_values=[0.0, 0.0]):
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
                                        position_trajectory[i], forces=[200] * 2)
            rospy.sleep(duration/num_steps)