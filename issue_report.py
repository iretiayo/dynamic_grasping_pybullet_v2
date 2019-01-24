## Memory leaks problem goes away
## Do not ask me why...


import pybullet as p
import time
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])

# NOTE, without real time simulation and stop at a break point- memory leaks
p.setRealTimeSimulation(1)

# boxId = p.loadURDF("r2d2.urdf",cubeStartPos, cubeStartOrientation)

mico = p.loadURDF("/home/jxu/model.urdf")
cube = p.loadURDF("cube_small.urdf", [0.3, 0.3, 0])

# import ipdb; ipdb.set_trace()


# for i in range (10000):
#     p.stepSimulation()
#     time.sleep(1./240.)
if __name__ == "__main__":

    while 1:
        import time
        time.sleep(1)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(mico)
    # print(cubePos,cubeOrn)
    # p.disconnect()


### Guess 2
import pybullet as p
import pybullet_data
physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
cube = p.loadURDF("cube_small.urdf")
import ipdb; ipdb.set_trace()
