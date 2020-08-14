import pprint
from collections import OrderedDict
import os
import csv
import pybullet_data
import pybullet as p
import pybullet_utils as pu
import pandas as pd
from math import radians, cos, sin
import numpy as np
import random
from shapely.geometry import Polygon, Point
from trimesh import load_mesh


def write_csv_line(result_file_path, result):
    """ write a line in a csv file; create the file and write the first line if the file does not already exist """
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def configure_pybullet(rendering=False, debug=False, yaw=50.0, pitch=-35.0, dist=1.2, target=(0.0, 0.0, 0.0)):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    pu.reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)


def create_object_urdf(object_mesh_filepath, object_name,
                       urdf_template_filepath='assets/models/object_template.urdf',
                       urdf_target_object_filepath='assets/models/target_object.urdf'):
    # set_up urdf
    os.system('cp {} {}'.format(urdf_template_filepath, urdf_target_object_filepath))
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name.obj', object_mesh_filepath, urdf_target_object_filepath)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name', object_name, urdf_target_object_filepath)
    os.system(sed_cmd)
    return urdf_target_object_filepath


def get_candidate_indices(prior_csv_file_path, prior_success_rate):
    df = pd.read_csv(prior_csv_file_path, index_col=0)
    df_success = df.loc[df['success_rate'] >= prior_success_rate]
    return list(df_success.index)


def calculate_target_pose(start_pose_in_world, angle, distance):
    """ calculate end pose after translating for a distance (in meter) in the direction of angle (in degrees) """
    start_pose_in_object= [[0, 0, 0], [0, 0, 0, 1]]
    target_x = cos(radians(angle)) * distance
    target_y = sin(radians(angle)) * distance
    target_pose_in_object = [[target_x, target_y, 0], [0, 0, 0, 1]]
    target_pose_in_world = tfc.toMatrix(tfc.fromTf(start_pose_in_world)).dot(
        tfc.toMatrix(tfc.fromTf(target_pose_in_object)))
    target_pose_in_world = tfc.toTf(tfc.fromMatrix(target_pose_in_world))
    target_pose_in_world = [list(target_pose_in_world[0]), list(target_pose_in_world[1])]
    return target_pose_in_object, target_pose_in_world


def random_point_in_polygon(polygon):
    min_x, min_y, max_x, max_y = polygon.bounds

    x = random.uniform(min_x, max_x)
    y = random.uniform(min_y, max_y)

    while not Point([x, y]).within(polygon):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)

    return (x, y)


def load_object(object_name, xy_position, surface_height, rpy):
    """" rpy is in degrees """
    mesh_dir = 'assets/models/'
    rpy = [radians(i) for i in rpy]
    object_mesh_filepath = os.path.join(mesh_dir, '{}'.format(object_name), '{}.obj'.format(object_name))
    target_urdf = create_object_urdf(object_mesh_filepath, object_name)
    target_mesh = load_mesh(object_mesh_filepath)
    target_z = -target_mesh.bounds.min(0)[2] + surface_height
    target_initial_pose = [
        [xy_position[0], xy_position[1], target_z], pu.quaternion_from_euler(rpy)]
    return p.loadURDF(target_urdf,
                      basePosition=target_initial_pose[0],
                      baseOrientation=target_initial_pose[1])