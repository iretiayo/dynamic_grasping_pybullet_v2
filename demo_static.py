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
import rospkg

from reachability_utils.process_reachability_data_from_csv import load_reachability_data_from_dir


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--grasp_database_path', type=str, default='yeah')
    parser.add_argument('--disable_gui', action='store_true', default=False)
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('assets/models')
    args.robot_urdf = os.path.abspath('assets/mico/mico.urdf')

    args.reachability_data_dir = os.path.join(rospkg.RosPack().get_path('mico_reachability_config'), 'data')

    return args


def get_reachability_space(reachability_data_dir):
    rospy.loginfo("start creating sdf reachability space...")
    start_time = time.time()
    _, mins, step_size, dims, sdf_reachability_space = load_reachability_data_from_dir(reachability_data_dir)
    rospy.loginfo("sdf reachability space created, which takes {}".format(time.time()-start_time))
    return sdf_reachability_space, mins, step_size, dims


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


class DynamicGrasping:
    def __init__(self, target_id, target_name, grasp_database_path, reachability_data_dir):
        self.target_name = target_name
        self.target_id = target_id
        self.grasp_database_path = grasp_database_path
        self.grasp_database = np.load(os.path.join(self.grasp_database_path, self.target_name + '.npy'))
        self.reachability_data_dir = reachability_data_dir
        self.sdf_reachability_space, self.mins, self.step_size, self.dims = get_reachability_space(self.reachability_data_dir)

        print('here')

    def dynamic_grasp(self):
        while not False:
            target_pose = pu.get_body_pose(self.target_id)
            predicted_pose = target_pose
            planning_time, grasp = self.plan_grasp(target_pose)
            # plan_motion_time, motion =

    def plan_grasp(self, target_pose):
        start_time = time.time()
        grasps_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in self.grasp_database]
        sdf_values = gu.get_reachability_of_grasps_pose_2d(grasps_in_world,
                                                           self.sdf_reachability_space,
                                                           self.mins,
                                                           self.step_size,
                                                           self.dims)
        grasp_order_idxs = np.argsort(sdf_values)[::-1]
        planned_grasp = grasps_in_world[grasp_order_idxs[0]]
        # planned_grasp = gu.change_end_effector_link(gu.list_2_pose(planned_grasp), gu.link6_reference_to_ee)
        planning_time = time.time() - start_time
        print("Planning a grasp takes {:.6f}".format(planning_time))

        # grasps_in_world_ee = [gu.pose_2_list(gu.change_end_effector_link(gu.list_2_pose(g), gu.link6_reference_to_ee)) for g in grasps_in_world]
        grasps_in_world_ee = [gu.change_end_effector_link_pose_2d(g) for g in grasps_in_world]
        gu.visualize_grasps_with_reachability(grasps_in_world_ee, sdf_values)

        return planning_time, planned_grasp



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

    def step(self, time, arm_motion):
        # calculate conveyor pose, change constraint
        # calculate arm pose, control array
        self.robot.step(time, arm_motion)
        p.stepSimulation()


if __name__ == "__main__":
    args = get_args()
    configure_pybullet(args.disable_gui)

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]
    target_initial_pose = [[0.3, 0.3, -target_mesh.bounds.min(0)[2] + 0.02], [0, 0, 0, 1]]
    robot_initial_pose = [[0, 0, 0], [0, 0, 0, 1]]
    conveyor_initial_pose = [[0.3, 0.3, 0.01], [0, 0, 0, 1]]

    world = World(target_initial_pose, robot_initial_pose, conveyor_initial_pose, args.robot_urdf, target_urdf)
    dynamic_grasping_controller = DynamicGrasping(target_id=world.target,
                                                  target_name=args.object_name,
                                                  grasp_database_path=args.grasp_database_path,
                                                  reachability_data_dir=args.reachability_data_dir)
    dynamic_grasping_controller.dynamic_grasp()

    print("finished")

