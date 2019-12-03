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

""" Use Graspit as backend to generate grasps and test in pybullet,
    saved as pose lists in link6 reference link"""


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--load_folder_path', type=str, required=True)
    parser.add_argument('--save_folder_path', type=str, required=True)
    parser.add_argument('--num_grasps', type=int, default=1000)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--disable_gui', action='store_true', default=False)
    parser.add_argument('--min_success_rate', type=float, default=1)
    parser.add_argument('--back_off', type=float, default=0.05)
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('assets/models')
    args.gripper_urdf = os.path.abspath('assets/robotiq_2f_85_hand/robotiq_arg2f_85_model.urdf')

    args.save_folder_path = os.path.join(args.save_folder_path, args.object_name)
    if not os.path.exists(args.save_folder_path):
        os.makedirs(args.save_folder_path)

    return args


def create_object_urdf(object_mesh_filepath, object_name,
                       urdf_template_filepath='assets/object_template.urdf',
                       urdf_target_object_filepath='assets/target_object.urdf'):
    # set_up urdf
    os.system('cp {} {}'.format(urdf_template_filepath, urdf_target_object_filepath))
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name.obj', object_mesh_filepath, urdf_target_object_filepath)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name', object_name, urdf_target_object_filepath)
    os.system(sed_cmd)
    return urdf_target_object_filepath


class Controller:
    EEF_LINK_INDEX = 0
    GRIPPER_JOINT_NAMES = ['finger_joint', 'left_inner_knuckle_joint', 'left_inner_finger_joint',
                           'right_outer_knuckle_joint', 'right_inner_knuckle_joint', 'right_inner_finger_joint']
    OPEN_POSITION = [0] * 6
    CLOSED_POSITION = 0.72 * np.array([1, 1, -1, 1, 1, -1])
    BASELINK_COM = [0.000000, -0.000097, 0.035953]
    BASELINK_TO_COM = [[0.000000, -0.000097, 0.035953], [0, 0, 0, 1]]
    GRASPIT_LINK_TO_MOVEIT_LINK = [[0, 0, 0], [0, 0, 0, 1]]
    GRASPIT_LINK_TO_PYBULLET_LINK = ([0.0, 0.0, 0.0], [0.0, 0.706825181105366, 0.0, 0.7073882691671998])
    LIFT_VALUE = 0.2

    JOINT_INDICES_DICT = {}

    MAX_FINGER_BASE_JOINT = 2.44
    MAX_FINGER_TIP_JOINT = 0.84
    PROXIMAL_TIP_RATIO = MAX_FINGER_BASE_JOINT / MAX_FINGER_TIP_JOINT

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.cid = None
        joint_infos = [p.getJointInfo(robot_id, joint_index) for joint_index in range(p.getNumJoints(robot_id))]
        self.JOINT_INDICES_DICT = {entry[1]: entry[0] for entry in joint_infos}
        self.GRIPPER_INDICES = [self.JOINT_INDICES_DICT[name] for name in self.GRIPPER_JOINT_NAMES]

        self.all_joints = range(p.getNumJoints(self.robot_id))

    def reset_to(self, pose):
        """ the pose is for the link6 center of mass """
        p.resetBasePositionAndOrientation(self.robot_id, pose[0], pose[1])
        self.move_to(pose)

    def move_to(self, pose):
        """ the pose is for the link6 center of mass """
        num_steps = 240
        current_pose = self.get_pose()
        positions = np.linspace(current_pose[0], pose[0], num_steps)
        quaternions = np.linspace(current_pose[1], pose[1], num_steps)

        if self.cid is None:
            self.cid = p.createConstraint(parentBodyUniqueId=self.robot_id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                          childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=current_pose[0],
                                          childFrameOrientation=current_pose[1])
        for pos, ori in zip(positions, quaternions):
            p.changeConstraint(self.cid, jointChildPivot=pos, jointChildFrameOrientation=ori)
            p.stepSimulation()
        pu.step()

    def get_current_baselink_pose(self):
        ee_com_pose_2d = pu.get_link_pose(self.robot_id, -1)
        ee_pose_2d = tf_conversions.toTf(
            tf_conversions.fromTf(ee_com_pose_2d) * tf_conversions.fromTf(self.BASELINK_TO_COM).Inverse())

        return ee_com_pose_2d, ee_pose_2d

    def execute_grasp(self, graspit_pose_msg, back_off):
        """ High level grasp interface using grasp 2d in world frame (link6_reference_frame)"""
        baselink_reference_to_com = (np.array(self.BASELINK_COM), [0.0, 0.0, 0.0, 1.0])
        baselink_com_pose_msg = gu.change_end_effector_link(graspit_pose_msg, baselink_reference_to_com)
        pregrasp_baselink_com_pose_msg = gu.back_off(baselink_com_pose_msg, -back_off)

        # move to pre-grasp
        self.reset_to(tf_conversions.toTf(tf_conversions.fromMsg(pregrasp_baselink_com_pose_msg)))

        # move to grasp
        self.move_to(tf_conversions.toTf(tf_conversions.fromMsg(baselink_com_pose_msg)))

        # close and lift
        self.close_gripper()
        self.lift()

        # robust test
        self.shake(0.05)
        self.lift(0.2)
        self.lift(-0.2)
        pu.step(2)

    def execute_grasp_simple(self, graspit_pose_msg):
        """ High level grasp interface using graspit pose in world frame (link6_reference_frame)"""
        baselink_reference_to_com = (np.array(self.BASELINK_COM), [0.0, 0.0, 0.0, 1.0])
        baselink_com_pose_msg = gu.change_end_effector_link(graspit_pose_msg, baselink_reference_to_com)
        self.reset_to(tf_conversions.toTf(tf_conversions.fromMsg(baselink_com_pose_msg)))
        self.close_gripper()
        self.lift()

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

    def get_gripper_joint_values(self):
        return [p.getJointState(self.robot_id, self.JOINT_INDICES_DICT[name])[0] for name in self.GRIPPER_JOINT_NAMES]

    def open_gripper(self):
        num_steps = 240
        current_gripper_joint = self.get_gripper_joint_values()
        waypoints = np.linspace(current_gripper_joint, self.OPEN_POSITION, num_steps)
        for wp in waypoints:
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                        jointIndices=self.GRIPPER_INDICES,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=wp,
                                        # positionGains=[self.grasp_joints_position_gains_during_inc_grasp]*len(hand_joint_values),
                                        forces=[500] * len(wp)
                                        )
            p.stepSimulation()
        pu.step()

    def close_gripper(self):
        num_steps = 240
        waypoints = np.linspace(self.OPEN_POSITION, self.CLOSED_POSITION, num_steps)
        for wp in waypoints:
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                        jointIndices=self.GRIPPER_INDICES,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=wp,
                                        # positionGains=[self.grasp_joints_position_gains_during_inc_grasp]*len(hand_joint_values),
                                        forces=[100] * len(wp)
                                        )
            p.stepSimulation()
        pu.step()

    def lift(self, z=LIFT_VALUE):
        target_pose = self.get_pose()
        target_pose[0][2] += z
        self.move_to(target_pose)

    def shake(self, z=LIFT_VALUE):
        left_pose = self.get_pose()
        right_pose = self.get_pose()

        left_pose[0][0] -= z
        right_pose[0][0] += z
        for i in range(1):
            self.move_to(left_pose)
            self.move_to(right_pose)

    def get_pose(self):
        "the pose is for the link6 center of mass"
        return [list(p.getBasePositionAndOrientation(self.robot_id)[0]),
                list(p.getBasePositionAndOrientation(self.robot_id)[1])]


class World:

    def __init__(self, target_initial_pose, gripper_initial_pose, gripper_urdf, target_urdf):
        self.target_initial_pose = target_initial_pose
        self.gripper_initial_pose = gripper_initial_pose
        self.gripper_urdf = gripper_urdf
        self.target_urdf = target_urdf

        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])
        self.robot = p.loadURDF(self.gripper_urdf, self.gripper_initial_pose[0], self.gripper_initial_pose[1],
                                flags=p.URDF_USE_SELF_COLLISION)

        p.changeDynamics(self.target, -1, mass=0.5)
        for joint in range(p.getNumJoints(self.robot)):
            p.changeDynamics(self.robot, joint, mass=1)
        p.changeDynamics(self.robot, -1, mass=50)

        self.controller = Controller(self.robot)

    def reset(self):
        p.resetBasePositionAndOrientation(self.target, self.target_initial_pose[0], self.target_initial_pose[1])
        p.resetBasePositionAndOrientation(self.robot, self.gripper_initial_pose[0], self.gripper_initial_pose[1])
        self.controller.reset_to(self.gripper_initial_pose)
        self.controller.open_gripper()


if __name__ == "__main__":
    args = get_args()
    if args.disable_gui:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    pu.reset_camera(yaw=50.0, pitch=-35.0, dist=1.2)

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]
    target_initial_pose = [[0, 0, -target_mesh.bounds.min(0)[2] + 0.01], [0, 0, 0, 1]]
    gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]

    world = World(target_initial_pose, gripper_initial_pose, args.gripper_urdf, target_urdf)

    raw_graspit_grasps_in_object = np.load(os.path.join(args.load_folder_path, args.object_name + '.npy'))
    good_graspit_grasps_in_object = []

    num_successful_grasps = 0
    progressbar = tqdm.tqdm(total=len(raw_graspit_grasps_in_object))
    world.reset()
    object_pose = p.getBasePositionAndOrientation(world.target)
    success_height_threshold = object_pose[0][2] + world.controller.LIFT_VALUE - 0.05
    for grasp_id, graspit_grasp_in_object in enumerate(raw_graspit_grasps_in_object):
        successes = []
        g_link6_ref_in_object = pu.split_7d(graspit_grasp_in_object)
        g_link6_ref_in_world = gu.convert_grasp_in_object_to_world(object_pose, g_link6_ref_in_object)
        pu.create_frame_marker(g_link6_ref_in_world)  # for visualization

        graspit_grasp_pose_in_world = tf_conversions.toMsg(tf_conversions.fromTf(g_link6_ref_in_world))
        pybullet_grasp_pose_in_world = gu.change_end_effector_link(graspit_grasp_pose_in_world,
                                                                   world.controller.GRASPIT_LINK_TO_PYBULLET_LINK)

        for t in range(args.num_trials):  # test a single grasp
            world.controller.execute_grasp(pybullet_grasp_pose_in_world, args.back_off)
            success = p.getBasePositionAndOrientation(world.target)[0][2] > success_height_threshold
            successes.append(success)
            world.reset()
        success_rate = np.average(successes)
        num_successful_trials = np.count_nonzero(successes)
        if success_rate >= args.min_success_rate:
            num_successful_grasps += 1
            good_graspit_grasps_in_object.append(graspit_grasp_in_object)

        progressbar.update(1)
        progressbar.set_description("object name: {} | success rate {}/{} | overall success rate {}/{}".
                                    format(args.object_name, num_successful_trials, args.num_trials,
                                           num_successful_grasps, grasp_id))
    progressbar.close()
    np.save(os.path.join(args.save_folder_path, 'grasps_eef.npy'), np.array(good_graspit_grasps_in_object))
    print("finished")
