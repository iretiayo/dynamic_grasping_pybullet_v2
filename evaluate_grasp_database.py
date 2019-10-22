from __future__ import division
import os
import argparse
import numpy as np
import pandas as pd
import tqdm
import pybullet as p
import pybullet_data
from grasp_evaluation_eef_only import World, create_object_urdf
import trimesh
import pybullet_utils as pu
import grasp_utils as gu


""" Given a grasp database, evaluate the success rate of each object """


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--grasp_database', type=str, required=True)
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('assets/models')
    args.gripper_urdf = os.path.abspath('assets/mico_hand/mico_hand.urdf')

    return args


if __name__ == "__main__":
    args = get_args()
    p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    object_grasp_npy_list = os.listdir(args.grasp_database)
    object_names = [s.split('.')[0] for s in object_grasp_npy_list]
    for obj, obj_grasps_npy in zip(object_names, object_grasp_npy_list):
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(obj), '{}.obj'.format(obj))
        object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
        target_urdf = create_object_urdf(object_mesh_filepath, obj)
        target_mesh = trimesh.load_mesh(object_mesh_filepath)
        floor_offset = target_mesh.bounds.min(0)[2]
        target_initial_pose = [[0, 0, -target_mesh.bounds.min(0)[2] + 0.01], [0, 0, 0, 1]]
        gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]

        world = World(target_initial_pose, gripper_initial_pose, args.gripper_urdf, target_urdf)
        link6_reference_to_ee = ([0.0, 0.0, -0.16], [1.0, 0.0, 0.0, 0])
        ee_to_link6_reference = ([0.0, -3.3091697137634315e-14, -0.16], [-1.0, 0.0, 0.0, -1.0341155355510722e-13])

        successes = []
        grasp_database = np.load(os.path.join(args.grasp_database, obj_grasps_npy))
        bar = tqdm.tqdm(total=len(grasp_database))
        for grasp_in_object_link6_ref in grasp_database:
            world.reset()
            object_pose = p.getBasePositionAndOrientation(world.target)
            grasp_in_object_link6_ref = [grasp_in_object_link6_ref[:3], grasp_in_object_link6_ref[3:]]
            grasp_in_world_link6_ref = gu.convert_grasp_in_object_to_world(object_pose, grasp_in_object_link6_ref)
            world.controller.execute_grasp(gu.list_2_pose(grasp_in_world_link6_ref))
            success = p.getBasePositionAndOrientation(world.target)[0][2] > 0.2
            successes.append(success)
            bar.update(1)
            bar.set_description('{} | {:.4f}'.format(obj, np.count_nonzero(successes)/len(successes)))
        bar.close()