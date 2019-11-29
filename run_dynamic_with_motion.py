import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import trimesh
import argparse
import grasp_utils as gu
import misc_utils as mu
import pybullet_utils as pu
from collections import OrderedDict
import csv
import tqdm
import tf_conversions
from mico_controller import MicoController
import rospy
import rospkg
import threading
from geometry_msgs.msg import PoseStamped
from math import pi
import pprint
from dynamic_grasping_world import DynamicGraspingWorld


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--grasp_database_path', type=str, required=True)
    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--realtime', action='store_true', default=False)
    parser.add_argument('--num_trials', type=int, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--max_check', type=int, default=1)
    parser.add_argument('--back_off', type=float, default=0.05)
    parser.add_argument('--disable_reachability', action='store_true', default=False)
    parser.add_argument('--conveyor_speed', type=float, default=0.01)
    args = parser.parse_args()

    if args.realtime:
        args.rendering = True

    args.mesh_dir = os.path.abspath('assets/models')
    args.robot_urdf = os.path.abspath('assets/mico/mico.urdf')
    args.conveyor_urdf = os.path.abspath('assets/conveyor.urdf')

    args.reachability_data_dir = os.path.join(rospkg.RosPack().get_path('mico_reachability_config'), 'data')
    # timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    # args.runstr = 'static-'+timestr
    # args.result_dir = os.path.join(args.result_dir, args.runstr)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    args.result_file_path = os.path.join(args.result_dir, args.object_name + '.csv')
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



if __name__ == "__main__":
    args = get_args()
    mu.configure_pybullet(args.rendering)
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
    floor_offset = target_mesh.bounds.min(0)[2]
    target_z = -target_mesh.bounds.min(0)[2] + 0.02
    target_initial_pose = [[0.3, 0.3, target_z], [0, 0, 0, 1]]
    robot_initial_pose = [[0, 0, 0], [0, 0, 0, 1]]
    conveyor_initial_pose = [[0.3, 0.3, 0.01], [0, 0, 0, 1]]

    dynamic_grasping_world = DynamicGraspingWorld(target_name=args.object_name,
                                                  target_initial_pose=target_initial_pose,
                                                  robot_initial_pose=robot_initial_pose,
                                                  robot_initial_state=MicoController.HOME,
                                                  conveyor_initial_pose=conveyor_initial_pose,
                                                  robot_urdf=args.robot_urdf,
                                                  conveyor_urdf=args.conveyor_urdf,
                                                  conveyor_speed=args.conveyor_speed,
                                                  target_urdf=target_urdf,
                                                  target_mesh_file_path=object_mesh_filepath,
                                                  grasp_database_path=args.grasp_database_path,
                                                  reachability_data_dir=args.reachability_data_dir,
                                                  realtime=args.realtime,
                                                  max_check=args.max_check,
                                                  disable_reachability=args.disable_reachability,
                                                  back_off=args.back_off)

    for i in range(args.num_trials):
        distance, theta, length = dynamic_grasping_world.reset(mode='dynamic_linear')
        print(distance, theta, length)
        time.sleep(2)  # for moveit to update scene, might not be necessary, depending on computing power
        success, grasp_idx, dynamic_grasping_time = dynamic_grasping_world.dynamic_grasp()
        result = [('exp_idx', i),
                  ('grasp_idx', grasp_idx),
                  ('success', success),
                  ('dynamic_grasping_time', dynamic_grasping_time),
                  ('theta', theta),
                  ('length', length),
                  ('distance', distance)]
        mu.write_csv_line(result_file_path=args.result_file_path, result=result)
