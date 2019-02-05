""" start moving conveyor and also update the scene (target + conveyor) """
## having two shared memory causes one of them stopped after some time

import time
import rospy
import pybullet as p
import mico_moveit
import utils as ut

p.connect(p.SHARED_MEMORY)
mico_moveit = mico_moveit.MicoMoveit()

MOVE = True # if not MOVE, just update scene

if __name__ == "__main__":

    rospy.init_node("update_scene")
    mico_moveit.clear_scene()
    time.sleep(1) # need some time for clear scene to finish, otherwise floor does not show up
    mico_moveit.add_box("floor", ((0, 0, -0.005), (0, 0, 0, 1)), size=(2, 2, 0.01))

    cube = ut.get_body_id("cube_small_modified")
    conveyor = ut.get_body_id("conveyor")

    # distance along x to travel
    max_x = 0.3
    min_x = -0.3
    step = 0.003 # meters per 0.1 seconds

    pivot = ut.get_body_pose(conveyor)[0]

    if MOVE:
        cid = p.createConstraint(parentBodyUniqueId=conveyor, parentLinkIndex=-1, childBodyUniqueId=-1,
                                 childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                 parentFramePosition=[0, 0, 0], childFramePosition=pivot)
        direction = '+'
        try:
            while True:
                print(direction)
                print(pivot[0])
                if direction == "+":
                    pivot[0] = pivot[0] + step
                else:
                    pivot[0] = pivot[0] - step
                # p.changeConstraint(cid, pivot, jointChildFrameOrientation=orn, maxForce=50)
                p.changeConstraint(cid, pivot, maxForce=5000)
                if pivot[0] > max_x:
                    direction = "-"
                elif pivot[0] < min_x:
                    direction = "+"
                mico_moveit.add_box("cube", p.getBasePositionAndOrientation(cube), size=(0.05, 0.05, 0.05))
                mico_moveit.add_box("conveyor", p.getBasePositionAndOrientation(conveyor), size=(.1, .1, .02))
                time.sleep(.1)
        except KeyboardInterrupt:
            print("interrupt")
            ut.remove_all_constraints()
    else:
        while True:
            mico_moveit.add_box("cube", p.getBasePositionAndOrientation(cube), size=(0.05, 0.05, 0.05))
            mico_moveit.add_box("conveyor", p.getBasePositionAndOrientation(conveyor), size=(.1, .1, .02))
            time.sleep(.1)


