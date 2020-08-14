import os
import pybullet as p
import time
import trimesh
import argparse
import grasp_utils as gu
import misc_utils as mu
import pprint
from dynamic_grasping_world import DynamicGraspingWorld
import json
import pandas as pd
import ast


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--robot_name', type=str, default='mico',
                        help="mico or ur5_robotiq")
    parser.add_argument('--motion_mode', type=str, default='dynamic_linear')
    parser.add_argument('--grasp_database_path', type=str, required=True)
    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--realtime', action='store_true', default=False)
    parser.add_argument('--num_trials', type=int, default=100)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--max_check', type=int, default=20)
    parser.add_argument('--back_off', type=float, default=0.05)
    parser.add_argument('--distance_low', type=float, default=0.15)
    parser.add_argument('--distance_high', type=float, default=0.4)
    parser.add_argument('--circular_distance_low', type=float, default=0.3)
    parser.add_argument('--circular_distance_high', type=float, default=0.5)
    parser.add_argument('--disable_reachability', action='store_true', default=False)
    parser.add_argument('--record_videos', action='store_true', default=False)
    parser.add_argument('--baseline_experiment_path', type=str, help='use motion path in this file for the run')
    parser.add_argument('--failure_only', action='store_true', default=False)
    parser.add_argument('--exp_id', type=int)
    parser.add_argument('--load_obstacles', action='store_true', default=False)
    parser.add_argument('--use_motion_aware', action='store_true', default=False)
    parser.add_argument('--motion_aware_model_path', type=str)

    # dynamic hyper parameter
    parser.add_argument('--conveyor_speed', type=float, default=0.01)
    parser.add_argument('--grasp_threshold', type=float, default=0.03)
    parser.add_argument('--lazy_threshold', type=float, default=0.3)
    parser.add_argument('--large_prediction_threshold', type=float, default=0.3)
    parser.add_argument('--small_prediction_threshold', type=float, default=0.1)
    parser.add_argument('--distance_travelled_threshold', type=float, default=0.1)
    parser.add_argument('--close_delay', type=float, default=0.5)
    parser.add_argument('--use_seed_trajectory', action='store_true', default=False)
    parser.add_argument('--use_previous_jv', action='store_true', default=False)
    parser.add_argument('--use_box', action='store_true', default=False)
    parser.add_argument('--use_baseline_method', action='store_true', default=False)
    parser.add_argument('--use_kf', action='store_true', default=False)
    parser.add_argument('--use_gt', action='store_true', default=False)
    parser.add_argument('--pose_freq', type=int, default=5)
    parser.add_argument('--approach_prediction', action='store_true', default=False)
    parser.add_argument('--approach_prediction_duration', type=float, default=1.0)
    parser.add_argument('--fix_motion_planning_time', type=float)
    parser.add_argument('--fix_grasp_ranking_time', type=float)
    args = parser.parse_args()

    if args.realtime:
        args.rendering = True

    args.mesh_dir = os.path.abspath('assets/models')
    args.conveyor_urdf = os.path.abspath('assets/conveyor.urdf')

    # timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    # args.runstr = 'static-'+timestr

    # create result folder
    args.result_dir = os.path.join(args.result_dir, args.object_name)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    args.result_file_path = os.path.join(args.result_dir, args.object_name + '.csv')

    # create a video folders
    if args.record_videos:
        args.video_dir = os.path.join(args.result_dir, 'videos')
        if not os.path.exists(args.video_dir):
            os.makedirs(args.video_dir)
    return args


if __name__ == "__main__":
    args = get_args()
    json.dump(vars(args), open(os.path.join(args.result_dir, args.object_name + '.json'), 'w'), indent=4)
    mu.configure_pybullet(args.rendering, debug=args.debug)

    print()
    print("Arguments:")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print('\n')

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = mu.create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    target_extents = target_mesh.bounding_box.extents.tolist()
    floor_offset = target_mesh.bounds.min(0)[2]
    target_z = -target_mesh.bounds.min(0)[2] + 0.02
    target_initial_pose = [[0.3, 0.3, target_z], [0, 0, 0, 1]]
    robot_initial_pose = [[0, 0, 0], [0, 0, 0, 1]]
    conveyor_initial_pose = [[0.3, 0.3, 0.01], [0, 0, 0, 1]]
    obstacle_names = ['tomato_soup_can', 'power_drill', 'bleach_cleanser']

    dynamic_grasping_world = DynamicGraspingWorld(motion_mode=args.motion_mode,
                                                  target_name=args.object_name,
                                                  obstacle_names=obstacle_names,
                                                  mesh_dir=args.mesh_dir,
                                                  robot_name=args.robot_name,
                                                  target_initial_pose=target_initial_pose,
                                                  conveyor_initial_pose=conveyor_initial_pose,
                                                  conveyor_urdf=args.conveyor_urdf,
                                                  conveyor_speed=args.conveyor_speed,
                                                  target_urdf=target_urdf,
                                                  target_mesh_file_path=object_mesh_filepath,
                                                  target_extents=target_extents,
                                                  grasp_database_path=args.grasp_database_path,
                                                  reachability_data_dir=None,
                                                  realtime=args.realtime,
                                                  max_check=args.max_check,
                                                  disable_reachability=args.disable_reachability,
                                                  back_off=args.back_off,
                                                  pose_freq=args.pose_freq,
                                                  use_seed_trajectory=args.use_seed_trajectory,
                                                  use_previous_jv=args.use_previous_jv,
                                                  use_kf=args.use_kf,
                                                  use_gt=args.use_gt,
                                                  grasp_threshold=args.grasp_threshold,
                                                  lazy_threshold=args.lazy_threshold,
                                                  large_prediction_threshold=args.large_prediction_threshold,
                                                  small_prediction_threshold=args.small_prediction_threshold,
                                                  close_delay=args.close_delay,
                                                  distance_travelled_threshold=args.distance_travelled_threshold,
                                                  distance_low=args.distance_low,
                                                  distance_high=args.distance_high,
                                                  circular_distance_low=args.circular_distance_low,
                                                  circular_distance_high=args.circular_distance_high,
                                                  use_box=args.use_box,
                                                  use_baseline_method=args.use_baseline_method,
                                                  approach_prediction=args.approach_prediction,
                                                  approach_prediction_duration=args.approach_prediction_duration,
                                                  fix_motion_planning_time=args.fix_motion_planning_time,
                                                  fix_grasp_ranking_time=args.fix_grasp_ranking_time,
                                                  load_obstacles=args.load_obstacles,
                                                  obstacle_distance_low=0.15,
                                                  obstacle_distance_high=0.25,
                                                  distance_between_region=0.05,
                                                  use_motion_aware=args.use_motion_aware,
                                                  motion_aware_model_path=args.motion_aware_model_path)

    # adding option to use previous experiment as config
    baseline_experiment_config_df = None
    if args.baseline_experiment_path and os.path.exists(args.baseline_experiment_path):
        args.baseline_experiment_path = os.path.join(args.baseline_experiment_path, args.object_name,
                                                     '{}.csv'.format(args.object_name))
        if os.path.exists(args.baseline_experiment_path):
            baseline_experiment_config_df = pd.read_csv(args.baseline_experiment_path, index_col=0)
            for key in ['target_quaternion', 'obstacle_poses']:
                if key in baseline_experiment_config_df.keys():
                    baseline_experiment_config_df[key] = baseline_experiment_config_df[key].apply(
                        lambda x: ast.literal_eval(x))
                    if key == 'obstacle_poses':
                        baseline_experiment_config_df[key] = baseline_experiment_config_df[key].apply(
                            lambda x: [x[p:p + 7] for p in range(0, len(x), 7)])
                else:
                    baseline_experiment_config_df[key] = None

            args.num_trials = len(baseline_experiment_config_df)

    for i in range(args.num_trials):
        if args.exp_id is not None and i != args.exp_id:
            continue
        reset_dict = None
        # reset_dict = {
        #     'distance': 0.2787919083152529,
        #     'length': 1.0,
        #     'theta': 110.23333162496952,
        #     'direction': 1,
        #     'target_quaternion': [0.0, 0.0, 0.8092568854035559, 0.5874549288472571]
        # }
        if baseline_experiment_config_df is not None:
            reset_dict = baseline_experiment_config_df.loc[i].to_dict()
            if args.failure_only and reset_dict['success']:
                print('skipping trial {}'.format(i))
                continue

        distance, theta, length, direction, target_quaternion, obstacle_poses = dynamic_grasping_world.reset(
            mode=args.motion_mode,
            reset_dict=reset_dict)
        time.sleep(2)  # for moveit to update scene, might not be necessary, depending on computing power
        if args.record_videos:
            logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join(args.video_dir, '{}.mp4'.format(i)))
        success, grasp_idx, dynamic_grasping_time = dynamic_grasping_world.dynamic_grasp()
        if args.record_videos:
            p.stopStateLogging(logging)
        result = [('exp_idx', i),
                  ('grasp_idx', grasp_idx),
                  ('success', success),
                  ('dynamic_grasping_time', dynamic_grasping_time),
                  ('theta', theta),
                  ('length', length),
                  ('distance', distance),
                  ('direction', direction),
                  ('target_quaternion', target_quaternion),
                  ('obstacle_poses', sum(obstacle_poses, []))]
        mu.write_csv_line(result_file_path=args.result_file_path, result=result)
