import os
import numpy as np
import pybullet as p
import trimesh
import argparse
import grasp_utils as gu
import pybullet_utils as pu
import tqdm
from eef_only_grasping_world import EEFOnlyDynamicWorld
import misc_utils as mu
from math import radians


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--grasp_database_path', type=str, required=True)
    parser.add_argument('--save_folder_path', type=str, required=True)
    parser.add_argument('--disable_gui', action='store_true', default=False)
    parser.add_argument('--back_off', type=float, default=0.05)

    parser.add_argument('--max_speed', type=float, default=0.05, help='maximum speed of the object')
    parser.add_argument('--min_speed', type=float, default=0.01, help='minimum speed of the object')
    parser.add_argument('--num_trials_per_grasp', type=int, default=1000, help='num trials for each grasp')

    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('assets/models')
    args.gripper_urdf = os.path.abspath('assets/mico/mico_hand.urdf')
    args.conveyor_urdf = os.path.abspath('assets/conveyor.urdf')

    args.save_folder_path = os.path.join(args.save_folder_path, args.object_name)
    if not os.path.exists(args.save_folder_path):
        os.makedirs(args.save_folder_path)
    args.result_file_path = os.path.join(args.save_folder_path, args.object_name + '.csv')

    return args


if __name__ == "__main__":
    args = get_args()
    rendering = not args.disable_gui
    mu.configure_pybullet(rendering=rendering, debug=False)

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = mu.create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]
    target_initial_pose = [[0, 0, -target_mesh.bounds.min(0)[2] + 0.01], [0, 0, 0, 1]]
    gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]
    conveyor_initial_pose = [[0, 0, 0.01], [0, 0, 0, 1]]

    world = EEFOnlyDynamicWorld(target_initial_pose,
                                conveyor_initial_pose,
                                gripper_initial_pose,
                                args.gripper_urdf,
                                target_urdf,
                                args.conveyor_urdf,
                                min_speed=args.min_speed,
                                max_speed=args.max_speed)

    grasps_eef, grasps_link6_ref, grasps_link6_com, pre_grasps_eef, pre_grasps_link6_ref, pre_grasps_link6_com = \
        gu.load_grasp_database(args.grasp_database_path, args.object_name, args.back_off)

    # 7D pregrasp pose, 7D grasp pose, 1 angle, 1 speed
    data = np.zeros((len(grasps_eef) * args.num_trials_per_grasp, 16))
    labels = np.zeros((len(grasps_eef) * args.num_trials_per_grasp, 1))
    pbar = tqdm.tqdm(total=len(grasps_eef) * args.num_trials_per_grasp)
    for i, (grasp_link6_com_in_object, pre_grasp_link6_com_in_object, grasp_eef, pre_grasp_eef) \
            in enumerate(zip(grasps_link6_com, pre_grasps_link6_com, grasps_eef, pre_grasps_eef)):
        # reset_dict = {'angle': 180, 'speed': 0.05, 'distance': 0.5}
        # world.reset(reset_dict)
        for t in range(args.num_trials_per_grasp):
            angle, speed = world.reset()
            success = world.dynamic_grasp(grasp_link6_com_in_object, pre_grasp_link6_com_in_object)
            data[i * args.num_trials_per_grasp + t] = list(pre_grasp_eef) + list(grasp_eef) + [radians(angle)] + [speed]
            labels[i * args.num_trials_per_grasp + t] = float(success)
            pbar.update(1)
            pbar.set_description('grasp: {}/{}'.format(i+1, len(grasps_eef)))
    pbar.close()
    # save data and labels
    np.save(os.path.join(args.save_folder_path, 'data.npy'), data)
    print('data.npy saved in {}'.format(os.path.join(args.save_folder_path, 'data.npy')))
    np.save(os.path.join(args.save_folder_path, 'labels.npy'), labels)
    print('labels.npy saved in {}'.format(os.path.join(args.save_folder_path, 'labels.npy')))