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


""" Use Graspit as backend to generate grasps and test in pybullet,
    saved as pose lists in link6 reference link"""


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--grasp_folder_path', type=str, default='grasp_dir',
                        help="Directory to store grasps and results. Ex: grasps_dir")
    parser.add_argument('--num_grasps', type=int, default=1000)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--disable_gui', action='store_true', default=False)
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('assets/models')
    args.robot_urdf = os.path.abspath('assets/mico/mico.urdf')

    args.grasp_folder_path = os.path.join(args.grasp_folder_path, args.object_name)
    args.result_file_path = os.path.join(args.grasp_folder_path, 'result.csv')
    if not os.path.exists(args.grasp_folder_path):
        os.makedirs(args.grasp_folder_path)

    return args


def configure_pybullet(disable_gui=False):
    if disable_gui:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)


def write_csv_line(result_file_path, index, num_trials, num_successes, volume_quality, epsilon_quality, grasp_fnm):
    result = [('index', index),
              ('num_trials', num_trials),
              ('num_successes', num_successes),
              ('volume_quality', volume_quality),
              ('epsilon_quality', epsilon_quality),
              ('grasp_fnm', grasp_fnm)]
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


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


def convert_grasp_in_object_to_world(object_pose, grasp_in_object):
    """
    :param object_pose: 2d list
    :param grasp_in_object: 2d list
    """
    object_T_grasp = tf_conversions.toMatrix(tf_conversions.fromTf(grasp_in_object))
    world_T_object = tf_conversions.toMatrix(tf_conversions.fromTf(object_pose))
    world_T_grasp = world_T_object.dot(object_T_grasp)
    grasp_in_world = tf_conversions.toTf(tf_conversions.fromMatrix(world_T_grasp))
    return grasp_in_world


def convert_grasp_in_world_to_object(object_pose, grasp_in_world):
    """
    :param object_pose: 2d list
    :param grasp_in_world: 2d list
    """
    world_T_object = tf_conversions.fromTf(object_pose)
    object_T_world = world_T_object.Inverse()
    object_T_world = tf_conversions.toMatrix(object_T_world)
    world_T_grasp = tf_conversions.fromTf(grasp_in_world)
    object_T_grasp = object_T_world.dot(world_T_grasp)
    grasp_in_object = object_T_grasp.toTf()
    return grasp_in_object


def dynamic_grasp():
    pass


class World:

    def __init__(self, target_initial_pose, robot_initial_pose, conveyor_initial_pose, robot_urdf, target_urdf):
        self.target_initial_pose = target_initial_pose
        self.robot_initial_pose = robot_initial_pose
        self.conveyor_initial_pose = conveyor_initial_pose
        self.robot_urdf = robot_urdf
        self.target_urdf = target_urdf

        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])
        self.robot = p.loadURDF(self.robot_urdf, self.robot_initial_pose[0], self.robot_initial_pose[1], flags=p.URDF_USE_SELF_COLLISION)
        self.conveyor = p.loadURDF("assets/conveyor.urdf", conveyor_initial_pose[0], conveyor_initial_pose[1])

        self.controller = MicoController(self.robot)
        self.reset()

    def reset(self):
        p.resetBasePositionAndOrientation(self.target, self.target_initial_pose[0], self.target_initial_pose[1])
        p.resetBasePositionAndOrientation(self.robot, self.robot_initial_pose[0], self.robot_initial_pose[1])
        p.resetBasePositionAndOrientation(self.robot, self.robot_initial_pose[0], self.robot_initial_pose[1])

        self.controller.set_arm_joints(MicoController.HOME)
        self.controller.control_arm_joints(MicoController.HOME)
        pu.step(2)

    def step(self):
        """ proceed the world by 1/240 second """
        # calculate conveyor pose, change constraint
        # calculate arm pose, control array
        p.stepSimulation()


if __name__ == "__main__":
    args = get_args()
    configure_pybullet(args.disable_gui)

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]
    target_initial_pose = [[0.3, 0, -target_mesh.bounds.min(0)[2] + 0.02], [0, 0, 0, 1]]
    robot_initial_pose = [[0, 0, 0], [0, 0, 0, 1]]
    conveyor_initial_pose = [[0.3, 0, 0.01], [0, 0, 0, 1]]

    world = World(target_initial_pose, robot_initial_pose, conveyor_initial_pose, args.robot_urdf, target_urdf)
    link6_reference_to_ee = ([0.0, 0.0, -0.16], [1.0, 0.0, 0.0, 0])
    ee_to_link6_reference = ([0.0, -3.3091697137634315e-14, -0.16], [-1.0, 0.0, 0.0, -1.0341155355510722e-13])

    print("finished")

