import pybullet as p
import time
import pybullet_data
from mico_controller import MicoController
from mico_moveit import MicoMoveit
import rospy
import graspit_commander
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
import pickle
import tf_conversions
import tf_manager
import tf
import tf2_ros
import tf2_kdl
import numpy as np
from math import pi
import tf.transformations as tft
import utils as ut

## TODO long box, not working
## TODO set the state of other joints for ik ***
## TODO make ik be able to tell why no ik ***
## TODO uniform sampling grasps


physicsClient = p.connect(p.GUI_SERVER)#or p.DIRECT for non-graphical version
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
ut.reset_camera(dist=1.5)
p.setAdditionalSearchPath("/home/jxu/bullet3/data") #optionally
# /home/jxu/.local/lib/python2.7/site-packages/pybullet_data
# /home/jxu/bullet3/examples/pybullet/examples

p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

## memory leaks happen sometimes without this but a breakpoint
p.setRealTimeSimulation(1)

mico = p.loadURDF("model/mico.urdf", flags=p.URDF_USE_SELF_COLLISION)
mc = MicoController(mico)
mc.reset_arm_joint_values(mc.HOME)
cube = p.loadURDF("model/cube_small_modified.urdf", [0, -0.5, 0.025+0.01])
conveyor = p.loadURDF("model/conveyor.urdf", [0, -0.5, 0.01])

j = mc.get_arm_joint_values()


# test pybullet
# while True:
#
#     j[5] += 0.01
#     print(j)
#     p.setJointMotorControlArray(mico, mc.GROUP_INDEX['arm'], p.POSITION_CONTROL, j,
#                                 forces=[2000] * len(mc.GROUP_INDEX['arm']))
#     time.sleep(0.01)

mico_moveit = MicoMoveit()
j = mico_moveit.arm_commander_group.get_current_joint_values()

while True:
    j[5] += 0.01
    print(mico_moveit.arm_commander_group.get_current_joint_values())
    plan = mico_moveit.arm_commander_group.plan(j)
    print(plan.joint_trajectory.points)
    mico_moveit.arm_commander_group.execute(plan)
    time.sleep(0.1)