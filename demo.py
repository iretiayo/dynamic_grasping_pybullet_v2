import pybullet as p
import time
import pybullet_data
from mico_controller import MicoController
import rospy

## TODO
## 2. arm hangs down after useing the correct IK to reset the arm from moveit

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

## memory leaks happen sometimes without this but a breakpoint
p.setRealTimeSimulation(1)

mico = p.loadURDF("/home/jxu/model.urdf")
mc = MicoController(mico)
mc.reset_arm_joint_values(mc.HOME)
cube = p.loadURDF("cube_small.urdf", [0, -0.5, 0])

test_eef_pose = [[-0.002576703886924381, -0.2696029068425193, 0.41288797205298017], [0.6823486760628231, -0.2289190614409086, 0.6884473485808099, 0.08964706250836511]]

if __name__ == "__main__":
    rospy.init_node("demo")

    j_v = mc.get_arm_ik(test_eef_pose)

    mc.move_arm_joint_values(mc.HOME)
    mc.move_arm_joint_values(j_v)


    # mc.reset_arm_joint_values(j_v)
    # print(mc.get_link_state(mc.ARM_EEF_INDEX))



    while 1:
        time.sleep(1)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(mico)
    # print(cubePos,cubeOrn)
    # p.disconnect()


