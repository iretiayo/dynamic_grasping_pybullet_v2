import pybullet as p
import time
import pybullet_data
from mico_controller import MicoController
import rospy

## TODO Finish open gripper and close gripper

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
# /home/jxu/.local/lib/python2.7/site-packages/pybullet_data

p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

## memory leaks happen sometimes without this but a breakpoint
p.setRealTimeSimulation(1)

mico = p.loadURDF("/home/jxu/model.urdf", flags=p.URDF_USE_SELF_COLLISION)
mc = MicoController(mico)
mc.reset_arm_joint_values(mc.HOME)
cube = p.loadURDF("model/cube_small_modified.urdf", [0, -0.3, 0.3])

test_eef_pose = [[-0.002576703886924381, -0.2696029068425193, 0.41288797205298017], [0.6823486760628231, -0.2289190614409086, 0.6884473485808099, 0.08964706250836511]]
pre_g_pose = ((-0.010223939612620055, -0.3314153914498819, 0.0738960532210591), (-0.6530495821195196, 0.6914430009073206, 0.1480400136529272, 0.27114013747035265))
g_pose = ((-0.0011438914860288536, -0.30347247297334357, 0.05343933520003911), (-0.6530495821195195, 0.6914430009073207, 0.14804001365292718, 0.27114013747035254))


if __name__ == "__main__":
    rospy.init_node("demo")

    j_v = mc.get_arm_ik(test_eef_pose)
    pre_g_joint_values = mc.get_arm_ik(pre_g_pose)
    g_joint_values = mc.get_arm_ik(g_pose)

    mc.move_arm_joint_values(mc.HOME)
    mc.open_gripper()

    ## everything is set up
    mc.move_arm_joint_values(j_v)
    mc.move_arm_joint_values(pre_g_joint_values)
    mc.move_arm_joint_values(g_joint_values)


    # mc.reset_arm_joint_values(j_v)
    # print(mc.get_link_state(mc.ARM_EEF_INDEX))

    mc.close_gripper()
    mc.move_arm_joint_values(mc.HOME)



    while 1:
        time.sleep(1)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(mico)
    # print(cubePos,cubeOrn)
    # p.disconnect()


