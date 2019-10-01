import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import trimesh
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('-o', '--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('-e', '--experiment_params_fname', type=str, default='experiment_params.yaml',
                        help="Config file for experiment params. Ex: experiment_params.yaml")
    parser.add_argument('-rd', '--result_dir', type=str, default='result_dir',
                        help="Directory to store results. Ex: result_dir")
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('dynamic_grasping_assets/models')
    args.gripper_urdf = os.path.abspath('dynamic_grasping_assets/mico_hand/mico_hand.urdf')
    return args


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

    def __init__(self, robot_id):
        self.robot_id = robot_id
        self.cid = None

    def move_to(self, pose):
        if self.cid is None:
            self.cid = p.createConstraint(parentBodyUniqueId=self.robot_id, parentLinkIndex=self.EEF_LINK_INDEX, childBodyUniqueId=-1,
                                     childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                     parentFramePosition=[0, 0, 0], childFramePosition=pose[0], childFrameOrientation=pose[1])
        else:
            p.changeConstraint(self.cid, jointChildPivot=pose[0], jointChildFrameOrientation=pose[1])
        step()

    def close_gripper(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.GRIPPER_INDICES,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.CLOSED_POSITION)
        step()

    def open_gripper(self):
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.GRIPPER_INDICES,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.OPEN_POSITION)
        step()


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
    target_urdf = create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    target_initial_pose = [[0, 0, -target_mesh.bounds.min(0)[2] + 0.01], [0, 0, 0, 1]]
    gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]

    world = World(target_initial_pose, gripper_initial_pose, args.gripper_urdf, target_urdf)

    for i in range(100):
        # start iterating grasps and evaluate
        world.reset()
        pass

    print("here")

