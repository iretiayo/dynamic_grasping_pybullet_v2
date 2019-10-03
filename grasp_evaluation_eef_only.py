import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import trimesh
import argparse
import grasp_utils as gu
import pybullet_helper as ph
from collections import OrderedDict
import csv


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('-o', '--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('-e', '--experiment_params_fname', type=str, default='experiment_params.yaml',
                        help="Config file for experiment params. Ex: experiment_params.yaml")
    parser.add_argument('-rd', '--grasp_dir', type=str, default='grasp_dir',
                        help="Directory to store grasps and results. Ex: grasps_dir")
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('dynamic_grasping_assets/models')
    args.gripper_urdf = os.path.abspath('dynamic_grasping_assets/mico_hand/mico_hand.urdf')

    args.grasp_dir = os.path.join(args.grasp_dir, args.object_name)
    args.result_file_path = os.path.join(args.grasp_dir, 'result.csv')
    if not os.path.exists(args.grasp_dir):
        os.makedirs(args.grasp_dir)
    return args


def write_csv_line(index, num_trials, success_rate, result_file_path):
    result = [('index', index),
              ('num_trials', num_trials),
              ('success_rate', success_rate)]
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def create_object_urdf(object_mesh_filepath, object_name,
                       urdf_template_filepath='model/object_template.urdf',
                       urdf_target_object_filepath='model/target_object.urdf'):
    # set_up urdf
    os.system('cp {} {}'.format(urdf_template_filepath, urdf_target_object_filepath))
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name.obj', object_mesh_filepath, urdf_target_object_filepath)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name', object_name, urdf_target_object_filepath)
    os.system(sed_cmd)
    return urdf_target_object_filepath


def step(duration=1):
    for i in range(duration*240):
        p.stepSimulation()


def step_real(duration=1):
    for i in range(duration*240):
        p.stepSimulation()
        time.sleep(1.0/240.0)


class Controller:
    EEF_LINK_INDEX = 0
    GRIPPER_INDICES = [1, 3]
    OPEN_POSITION = [0.0, 0.0]
    CLOSED_POSITION = [1.1, 1.1]
    LINK6_COM = [-0.002216, -0.000001, -0.058489]

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
                                     parentFramePosition=[0, 0, 0], childFramePosition=current_pose[0], childFrameOrientation=current_pose[1])
        for pos, ori in zip(positions, quaternions):
            p.changeConstraint(self.cid, jointChildPivot=pos, jointChildFrameOrientation=ori)
            p.stepSimulation()
        step()

    def close_gripper(self):
        num_steps = 240
        waypoints = np.linspace(self.OPEN_POSITION, self.CLOSED_POSITION, num_steps)
        for wp in waypoints:
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                        jointIndices=self.GRIPPER_INDICES,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=wp,
                                        forces=[10, 10]
                                        )
            p.stepSimulation()
        step()

    def execute_grasp(self, graspit_pose_msp):
        """ High level grasp interface using graspit pose in world frame (link6_reference_frame)"""
        link6_reference_to_link6_com = (self.LINK6_COM, [0.0, 0.0, 0.0, 1.0])
        link6_com_pose_msg = gu.change_end_effector_link(graspit_pose_msp, link6_reference_to_link6_com)
        self.reset_to(ph.pose_2_list(link6_com_pose_msg))
        self.close_gripper()
        self.lift()

    def open_gripper(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.GRIPPER_INDICES,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.OPEN_POSITION)
        step()

    def lift(self, z=0.2):
        target_pose = self.get_pose()
        target_pose[0][2] += z
        self.move_to(target_pose)

    def get_pose(self):
        "the pose is for the link6 center of mass"
        return [list(p.getBasePositionAndOrientation(self.robot_id)[0]), list(p.getBasePositionAndOrientation(self.robot_id)[1])]


class World:

    def __init__(self, target_initial_pose, gripper_initial_pose, gripper_urdf, target_urdf):
        self.target_initial_pose = target_initial_pose
        self.gripper_initial_pose = gripper_initial_pose
        self.gripper_urdf = gripper_urdf
        self.target_urdf = target_urdf

        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])
        self.robot = p.loadURDF(self.gripper_urdf, self.gripper_initial_pose[0], self.gripper_initial_pose[1], flags=p.URDF_USE_SELF_COLLISION)

        self.controller = Controller(self.robot)

    def reset(self):
        p.resetBasePositionAndOrientation(self.target, target_initial_pose[0], target_initial_pose[1])
        p.resetBasePositionAndOrientation(self.robot, gripper_initial_pose[0], gripper_initial_pose[1])
        self.controller.move_to(gripper_initial_pose)
        self.controller.open_gripper()


if __name__ == "__main__":
    args = get_args()
    p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    # p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=-24.4, cameraPitch=-47.0,
    #                              cameraTargetPosition=(-0.2, -0.30, 0.0))

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]
    target_initial_pose = [[0, 0, -target_mesh.bounds.min(0)[2] + 0.01], [0, 0, 0, 1]]
    gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]

    world = World(target_initial_pose, gripper_initial_pose, args.gripper_urdf, target_urdf)
    link6_reference_to_ee = ([0.0, 0.0, -0.16], [1.0, 0.0, 0.0, 0])
    ee_to_link6_reference = ([0.0, -3.3091697137634315e-14, -0.16], [-1.0, 0.0, 0.0, -1.0341155355510722e-13])

    for i in range(1000):
        # start sampling grasps and evaluate
        world.reset()
        object_pose = p.getBasePositionAndOrientation(world.target)
        object_pose_msg = ph.list_2_pose(object_pose)
        graspit_grasps, graspit_grasp_poses_in_world, graspit_grasp_poses_in_object \
            = gu.generate_grasps(object_mesh=object_mesh_filepath_ply,
                                 object_pose=object_pose_msg,
                                 uniform_grasp=False,
                                 floor_offset=floor_offset,
                                 max_steps=70000,
                                 save_fnm='grasps.pk',
                                 load_fnm='grasps.pk')
        for g_pose_msg in graspit_grasp_poses_in_world:
            num_trials = 10
            for _ in range(num_trials):  # test a single grasp
                world.controller.execute_grasp(g_pose_msg)
                success = p.getBasePositionAndOrientation(world.target)[0][2] > 0.1
                print(success)
                world.reset()
    print("here")

