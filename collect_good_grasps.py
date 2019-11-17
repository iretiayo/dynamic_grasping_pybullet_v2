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
    args.gripper_urdf = os.path.abspath('assets/mico/mico_hand.urdf')

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
    GRIPPER_INDICES = [1, 2, 3, 4]
    OPEN_POSITION = [0.0, 0.0, 0.0, 0.0]
    CLOSED_POSITION = [1.1, 0.0, 1.1, 0.0]
    LINK6_COM = [-0.002216, -0.000001, -0.058489]
    LIFT_VALUE = 0.2

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.cid = None

    def reset_to(self, pose):
        """ the pose is for the link6 center of mass """
        p.resetBasePositionAndOrientation(self.robot_id, pose[0], pose[1])
        self.move_to(pose)

    def move_to(self, pose):
        """ the pose is for the link6 center of mass """
        num_steps = 240
        current_pose = self.get_pose()
        positions = np.linspace(current_pose[0], pose[0], num_steps)
        angles = np.linspace(p.getEulerFromQuaternion(current_pose[1]), p.getEulerFromQuaternion(pose[1]), num_steps)
        quaternions = np.array([p.getQuaternionFromEuler(angle) for angle in angles])
        if self.cid is None:
            self.cid = p.createConstraint(parentBodyUniqueId=self.robot_id, parentLinkIndex=-1, childBodyUniqueId=-1,
                                          childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                          parentFramePosition=[0, 0, 0], childFramePosition=current_pose[0],
                                          childFrameOrientation=current_pose[1])
        for pos, ori in zip(positions, quaternions):
            p.changeConstraint(self.cid, jointChildPivot=pos, jointChildFrameOrientation=ori)
            p.stepSimulation()
        pu.step()

    def close_gripper(self):
        num_steps = 240
        waypoints = np.linspace(self.OPEN_POSITION, self.CLOSED_POSITION, num_steps)
        for wp in waypoints:
            pu.control_joints(self.robot_id, self.GRIPPER_INDICES, wp)
            p.stepSimulation()
        pu.step()

    def execute_grasp(self, grasp, back_off):
        """ High level grasp interface using grasp 2d in world frame (link6_reference_frame)"""
        link6_com_pose_2d = gu.change_end_effector_link_pose_2d(grasp, gu.link6_reference_to_link6_com)
        pre_link6_com_pose_2d = gu.back_off_pose_2d(link6_com_pose_2d, back_off)
        self.reset_to(pre_link6_com_pose_2d)
        actual_pre_ee_pose_2d = pu.get_link_pose(self.robot_id, 0)
        actual_pre_link6_ref_pose_2d = gu.change_end_effector_link_pose_2d(actual_pre_ee_pose_2d, gu.ee_to_link6_reference)
        actual_pre_link6_com_pose_2d = pre_link6_com_pose_2d

        self.move_to(link6_com_pose_2d)
        actual_ee_pose_2d = pu.get_link_pose(self.robot_id, 0)
        actual_link6_ref_pose_2d = gu.change_end_effector_link_pose_2d(actual_ee_pose_2d, gu.ee_to_link6_reference)
        actual_link6_com_pose_2d = link6_com_pose_2d
        self.close_gripper()
        self.lift()
        # robust test
        self.lift(0.2)
        self.lift(-0.2)
        pu.step(2)
        return actual_pre_ee_pose_2d, actual_pre_link6_ref_pose_2d, actual_pre_link6_com_pose_2d, actual_ee_pose_2d, actual_link6_ref_pose_2d, actual_link6_com_pose_2d

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
        pu.set_joint_positions(self.robot_id, self.GRIPPER_INDICES, self.OPEN_POSITION)
        pu.control_joints(self.robot_id, self.GRIPPER_INDICES, self.OPEN_POSITION)
        pu.step()

    def lift(self, z=LIFT_VALUE):
        target_pose = self.get_pose()
        target_pose[0][2] += z
        self.move_to(target_pose)

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

    grasps_link6_ref_in_object = np.load(os.path.join(args.load_folder_path, args.object_name + '.npy'))
    # placeholder to save good grasps
    grasps_eef = []
    grasps_link6_com = []
    grasps_link6_ref = []
    pre_grasps_eef = []
    pre_grasps_link6_com = []
    pre_grasps_link6_ref = []

    num_grasps = 0
    num_successful_grasps = 0
    progressbar = tqdm.tqdm(initial=num_grasps, total=args.num_grasps)
    while num_grasps < args.num_grasps:
        # start sampling grasps and evaluate
        world.reset()
        object_pose = p.getBasePositionAndOrientation(world.target)
        success_height_threshold = object_pose[0][2] + world.controller.LIFT_VALUE - 0.05
        for g_link6_ref_in_object in grasps_link6_ref_in_object:
            successes = []
            g_link6_ref_in_object = pu.split_7d(g_link6_ref_in_object)
            g_link6_ref_in_world = gu.convert_grasp_in_object_to_world(object_pose, g_link6_ref_in_object)
            # pu.create_frame_marker(g_link6_ref_in_world)    # for visualization
            for t in range(args.num_trials):  # test a single grasp
                actual_pre_ee_pose_2d, actual_pre_link6_ref_pose_2d, actual_pre_link6_com_pose_2d, actual_ee_pose_2d, actual_link6_ref_pose_2d, actual_link6_com_pose_2d\
                    = world.controller.execute_grasp(g_link6_ref_in_world, args.back_off)
                success = p.getBasePositionAndOrientation(world.target)[0][2] > success_height_threshold
                successes.append(success)
                # print(success)    # the place to put a break point
                world.reset()
            success_rate = np.average(successes)
            num_successful_trials = np.count_nonzero(successes)
            if success_rate >= args.min_success_rate:
                num_successful_grasps += 1

                grasp_eef_in_object = gu.convert_grasp_in_world_to_object(object_pose, actual_ee_pose_2d)
                grasp_link6_com_in_object = gu.convert_grasp_in_world_to_object(object_pose, actual_link6_com_pose_2d)
                grasp_link6_ref_in_object = gu.convert_grasp_in_world_to_object(object_pose, actual_link6_ref_pose_2d)
                pre_grasp_eef_in_object = gu.convert_grasp_in_world_to_object(object_pose, actual_pre_ee_pose_2d)
                pre_grasp_link6_com_in_object = gu.convert_grasp_in_world_to_object(object_pose, actual_pre_link6_com_pose_2d)
                pre_grasp_link6_ref_in_object = gu.convert_grasp_in_world_to_object(object_pose, actual_pre_link6_ref_pose_2d)

                grasps_eef.append(pu.merge_pose_2d(grasp_eef_in_object))
                grasps_link6_com.append(pu.merge_pose_2d(grasp_link6_com_in_object))
                grasps_link6_ref.append(pu.merge_pose_2d(grasp_link6_ref_in_object))
                pre_grasps_eef.append(pu.merge_pose_2d(pre_grasp_eef_in_object))
                pre_grasps_link6_com.append(pu.merge_pose_2d(pre_grasp_link6_com_in_object))
                pre_grasps_link6_ref.append(pu.merge_pose_2d(pre_grasp_link6_ref_in_object))

            num_grasps += 1
            progressbar.update(1)
            progressbar.set_description("object name: {} | success rate {}/{} | overall success rate {}/{}".
                                        format(args.object_name, num_successful_trials, args.num_trials,
                                               num_successful_grasps, num_grasps))
            if num_grasps == args.num_grasps:
                break
    progressbar.close()
    np.save(os.path.join(args.save_folder_path, 'grasps_eef.npy'), np.array(grasps_eef))
    np.save(os.path.join(args.save_folder_path, 'grasps_link6_com.npy'), np.array(grasps_link6_com))
    np.save(os.path.join(args.save_folder_path, 'grasps_link6_ref.npy'), np.array(grasps_link6_ref))
    np.save(os.path.join(args.save_folder_path, 'pre_grasps_eef_'+str(args.back_off)+'.npy'), np.array(pre_grasps_eef))
    np.save(os.path.join(args.save_folder_path, 'pre_grasps_link6_com_'+str(args.back_off)+'.npy'), np.array(pre_grasps_link6_com))
    np.save(os.path.join(args.save_folder_path, 'pre_grasps_link6_ref_'+str(args.back_off)+'.npy'), np.array(pre_grasps_link6_ref))
    print("finished")
