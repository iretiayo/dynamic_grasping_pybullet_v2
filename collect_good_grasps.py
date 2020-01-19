import os
import numpy as np
import pybullet as p
import pybullet_data
import trimesh
import argparse
import grasp_utils as gu
import pybullet_utils as pu
import tqdm
import misc_utils as mu
from eef_only_grasping_world import EEFOnlyStaticWorld


""" Load pregenerated raw graspit grasps, evaluate and then keep those that succeed over a threshold """


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--robot_config_name', type=str, default='mico',
                        help="name of robot configs to load from grasputils. Ex: mico or ur5_robotiq")
    parser.add_argument('--load_folder_path', type=str, required=True,
                        help="folder path to load raw grasps from graspit")
    parser.add_argument('--save_folder_path', type=str, required=True,
                        help="folder path to save collected good grasps")
    parser.add_argument('--num_successful_grasps', type=int, default=100)
    parser.add_argument('--num_grasps', type=int, default=5000)
    parser.add_argument('--num_trials', type=int, default=50)
    parser.add_argument('--disable_gui', action='store_true', default=False)
    parser.add_argument('--min_success_rate', type=float, default=0.95)
    parser.add_argument('--back_off', type=float, default=0.05)
    parser.add_argument('--apply_noise', action='store_true', default=False)
    parser.add_argument('--prior_folder_path', type=str,
                        help="folder path to the prior csv files")
    parser.add_argument('--prior_success_rate', type=float)
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('assets/models')

    args.save_folder_path = os.path.join(args.save_folder_path, args.object_name)
    if not os.path.exists(args.save_folder_path):
        os.makedirs(args.save_folder_path)
    args.result_file_path = os.path.join(args.save_folder_path, args.object_name+'.csv')

    if args.prior_folder_path is not None:
        if args.prior_success_rate is None:
            args.prior_success_rate = args.min_success_rate
        args.prior_csv_file_path = os.path.join(args.prior_folder_path, args.object_name, args.object_name+'.csv')
        args.candidate_indices = mu.get_candidate_indices(args.prior_csv_file_path, args.prior_success_rate)
    else:
        args.candidate_indices = []

    return args


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
    target_urdf = mu.create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]
    target_initial_pose = [[0, 0, -target_mesh.bounds.min(0)[2] + 0.01], [0, 0, 0, 1]]
    gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]

    world = EEFOnlyStaticWorld(target_initial_pose, gripper_initial_pose, args.robot_config_name, target_urdf, args.apply_noise)

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
    args.num_grasps = min(args.num_grasps, len(grasps_link6_ref_in_object))
    progressbar = tqdm.tqdm(initial=num_grasps, total=args.num_grasps)
    # start iterating grasps and evaluate
    world.reset()
    object_pose = p.getBasePositionAndOrientation(world.target)
    success_height_threshold = object_pose[0][2] + world.controller.LIFT_VALUE - 0.05
    for i, g_link6_ref_in_object in enumerate(grasps_link6_ref_in_object):
        if len(args.candidate_indices) != 0 and i not in args.candidate_indices and i < args.candidate_indices[-1]:
            progressbar.update(1)
            continue
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
        # write results
        result = [('grasp_index', i), ('success_rate', success_rate)]
        mu.write_csv_line(args.result_file_path, result)
        # have collected required number of grasps or have evaluated required number of grasps
        if num_successful_grasps == args.num_successful_grasps or num_grasps == args.num_grasps:
            break
    progressbar.close()
    np.save(os.path.join(args.save_folder_path, 'grasps_eef.npy'), np.array(grasps_eef))
    np.save(os.path.join(args.save_folder_path, 'grasps_link6_com.npy'), np.array(grasps_link6_com))
    np.save(os.path.join(args.save_folder_path, 'grasps_link6_ref.npy'), np.array(grasps_link6_ref))
    np.save(os.path.join(args.save_folder_path, 'pre_grasps_eef_'+str(args.back_off)+'.npy'), np.array(pre_grasps_eef))
    np.save(os.path.join(args.save_folder_path, 'pre_grasps_link6_com_'+str(args.back_off)+'.npy'), np.array(pre_grasps_link6_com))
    np.save(os.path.join(args.save_folder_path, 'pre_grasps_link6_ref_'+str(args.back_off)+'.npy'), np.array(pre_grasps_link6_ref))
    print("finished")
