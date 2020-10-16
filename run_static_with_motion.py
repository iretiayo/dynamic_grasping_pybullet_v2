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
from mico_controller import MicoController
import rospy
import threading
from geometry_msgs.msg import PoseStamped
from math import pi
import pprint
from dynamic_grasping_world import DynamicGraspingWorld


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--robot_config_name', type=str, default='mico',
                        help="name of robot configs to load from grasputils. Ex: mico or ur5_robotiq")
    parser.add_argument('--grasp_database_path', type=str, required=True)
    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--realtime', action='store_true', default=False)
    parser.add_argument('--num_trials', type=int, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--max_check', type=int, default=1)
    parser.add_argument('--back_off', type=float, default=0.05)
    parser.add_argument('--distance_low', type=float, default=0.15)
    parser.add_argument('--distance_high', type=float, default=0.4)
    parser.add_argument('--disable_reachability', action='store_true', default=False)
    parser.add_argument('--use_box', action='store_true', default=False)
    parser.add_argument('--use_baseline_method', action='store_true', default=False)
    parser.add_argument('--approach_prediction', action='store_true', default=False)
    parser.add_argument('--approach_prediction_duration', type=float, default=1.0)
    args = parser.parse_args()

    if args.realtime:
        args.rendering = True

    args.mesh_dir = os.path.abspath('assets/models')
    args.conveyor_urdf = os.path.abspath('assets/conveyor.urdf')

    robot_configs = gu.robot_configs[args.robot_config_name]
    args.__dict__.update(robot_configs.__dict__)

    # timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    # args.runstr = 'static-'+timestr
    # args.result_dir = os.path.join(args.result_dir, args.runstr)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    args.result_file_path = os.path.join(args.result_dir, args.object_name + '.csv')
    return args


def configure_pybullet(rendering=False):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    pu.reset_camera(yaw=50.0, pitch=-35.0, dist=1.200002670288086, target=(0.0, 0.0, 0.0))
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)


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


def write_csv_line(result_file_path,
                   exp_idx,
                   grasp_idx,
                   success,
                   grasp_attempted,
                   pre_grasp_reached,
                   grasp_reachaed,
                   grasp_planning_time,
                   num_ik_called,
                   target_position,
                   target_orientation,
                   distance,
                   comment):
    result = [('exp_idx', exp_idx),
              ('grasp_idx', grasp_idx),
              ('success', success),
              ('grasp_attempted', grasp_attempted),
              ('pre_grasp_reached', pre_grasp_reached),
              ('grasp_reachaed', grasp_reachaed),
              ('grasp_planning_time', grasp_planning_time),
              ('num_ik_called', num_ik_called),
              ('target_position', target_position),
              ('target_orientation', target_orientation),
              ('distance', distance),
              ('comment', comment)]
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


if __name__ == "__main__":
    args = get_args()
    configure_pybullet(args.rendering)
    rospy.init_node('dynamic_grasping')

    print()
    print("Arguments:")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(vars(args))
    print()

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    target_extents = target_mesh.bounding_box.extents.tolist()
    floor_offset = target_mesh.bounds.min(0)[2]
    conveyor_thickness = 0.02
    target_z = -target_mesh.bounds.min(0)[2] + conveyor_thickness
    target_initial_pose = [[0.3, 0.3, target_z], [0, 0, 0, 1]]
    robot_initial_pose = [[0, 0, 0], [0, 0, 0, 1]]
    conveyor_initial_pose = [[0.3, 0.3, conveyor_thickness/2], [0, 0, 0, 1]]

    dynamic_grasping_world = DynamicGraspingWorld(target_name=args.object_name,
                                                  obstacle_names=[],
                                                  mesh_dir=args.mesh_dir,
                                                  robot_config_name=args.robot_config_name,
                                                  target_initial_pose=target_initial_pose,
                                                  robot_initial_pose=robot_initial_pose,
                                                  robot_initial_state=MicoController.HOME,
                                                  conveyor_initial_pose=conveyor_initial_pose,
                                                  robot_urdf=args.robot_urdf,
                                                  conveyor_urdf=args.conveyor_urdf,
                                                  conveyor_speed=0,
                                                  target_urdf=target_urdf,
                                                  target_mesh_file_path=object_mesh_filepath,
                                                  target_extents=target_extents,
                                                  grasp_database_path=args.grasp_database_path,
                                                  reachability_data_dir=args.reachability_data_dir,
                                                  realtime=args.realtime,
                                                  max_check=args.max_check,
                                                  disable_reachability=args.disable_reachability,
                                                  back_off=args.back_off,
                                                  pose_freq=5,
                                                  use_seed_trajectory=None,
                                                  use_previous_jv=None,
                                                  use_gt=None,
                                                  use_kf=None,
                                                  use_lstm_prediction=None,
                                                  lstm_model_filepath=None,
                                                  grasp_threshold=None,
                                                  lazy_threshold=None,
                                                  large_prediction_threshold=None,
                                                  small_prediction_threshold=None,
                                                  close_delay=None,
                                                  distance_travelled_threshold=None,
                                                  distance_low=args.distance_low,
                                                  distance_high=args.distance_high,
                                                  circular_distance_low=args.distance_low,
                                                  circular_distance_high=args.distance_high,
                                                  use_box=args.use_box,
                                                  use_baseline_method=args.use_baseline_method,
                                                  approach_prediction=args.approach_prediction,
                                                  approach_prediction_duration=args.approach_prediction_duration,
                                                  fix_motion_planning_time=None,
                                                  fix_grasp_ranking_time=None,
                                                  load_obstacles=False,
                                                  obstacle_distance_low=0.15,
                                                  obstacle_distance_high=0.25,
                                                  distance_between_region=0.05,
                                                  use_motion_aware=False,
                                                  motion_aware_model_path=None)

    for i in range(args.num_trials):
        target_pose, distance = dynamic_grasping_world.reset(mode='static_random')
        # time.sleep(2)  # for moveit to update scene, might not be necessary, depending on computing power
        success, grasp_idx, grasp_attempted, pre_grasp_reached, grasp_reachaed, grasp_planning_time, num_ik_called, comment = \
            dynamic_grasping_world.static_grasp()
        write_csv_line(result_file_path=args.result_file_path,
                       exp_idx=i,
                       grasp_idx=grasp_idx,
                       success=success,
                       grasp_attempted=grasp_attempted,
                       pre_grasp_reached=pre_grasp_reached,
                       grasp_reachaed=grasp_reachaed,
                       grasp_planning_time=grasp_planning_time,
                       num_ik_called=num_ik_called,
                       target_position=target_pose[0],
                       target_orientation=target_pose[1],
                       distance=distance,
                       comment=comment)
