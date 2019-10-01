
import numpy as np
import pybullet as p
import pybullet_data
import time


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

    def __init__(self, target_initial_pose, gripper_initial_pose, gripper_urdf):
        self.target_initial_pose = target_initial_pose
        self.gripper_initial_pose = gripper_initial_pose
        self.gripper_urdf = gripper_urdf

        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF("cube_small.urdf", self.target_initial_pose[0], self.target_initial_pose[1])
        self.robot = p.loadURDF(self.gripper_urdf, self.gripper_initial_pose[0], self.gripper_initial_pose[1])

        self.controller = Controller(self.robot)

    def reset(self):
        p.resetBasePositionAndOrientation(self.target, target_initial_pose[0], target_initial_pose[1])
        p.resetBasePositionAndOrientation(self.robot, gripper_initial_pose[0], gripper_initial_pose[1])
        self.controller.move_to(gripper_initial_pose)


if __name__ == "__main__":
    p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    # p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=-24.4, cameraPitch=-47.0,
    #                              cameraTargetPosition=(-0.2, -0.30, 0.0))

    target_initial_pose = [[0, 0, 0], [0, 0, 0, 1]]
    gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]
    gripper_urdf = "mico_hand/mico_hand.urdf"

    world = World(target_initial_pose, gripper_initial_pose, gripper_urdf)

    for i in range(100):
        # start iterating grasps and evaluate
        world.reset()
        pass

    print("here")

