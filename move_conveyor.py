""" start moving conveyor and also update the scene (target + conveyor) """
## having two shared memory causes one of them stopped after some time

import time
import rospy
import pybullet as p
import mico_moveit
import utils as ut
import rospy
from geometry_msgs.msg import Pose

p.connect(p.SHARED_MEMORY)
mico_moveit = mico_moveit.MicoMoveit()
ut.remove_all_constraints()

MOVE = True # if not MOVE, just update scene

if __name__ == "__main__":

    rospy.init_node("update_scene")
    mico_moveit.clear_scene()
    time.sleep(1) # need some time for clear scene to finish, otherwise floor does not show up
    mico_moveit.add_box("floor", ((0, 0, -0.005), (0, 0, 0, 1)), size=(2, 2, 0.01))
    pub = rospy.Publisher('target_pose', Pose, queue_size=1)

    cube = ut.get_body_id("cube_small_modified")
    conveyor = ut.get_body_id("conveyor")

    speed = 0.02 # m/s

    # distance along x to travel
    max_x = 0.8
    min_x = -0.8
    step = speed/10.0 # meters per 0.1 seconds

    pivot = ut.get_body_pose(conveyor)[0]

    if MOVE:
        cid = p.createConstraint(parentBodyUniqueId=conveyor, parentLinkIndex=-1, childBodyUniqueId=-1,
                                 childLinkIndex=-1, jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                                 parentFramePosition=[0, 0, 0], childFramePosition=pivot)
        direction = '+' # for moving back and force
        c = time.time()
        print_speed = True  # print speed only once
        while True:
            # print(direction)
            # print(pivot[0])
            if direction == "+":
                pivot[0] = pivot[0] + step
            else:
                pivot[0] = pivot[0] - step
            # p.changeConstraint(cid, pivot, jointChildFrameOrientation=orn, maxForce=50)
            p.changeConstraint(cid, pivot, maxForce=5000)
            if pivot[0] > max_x and print_speed:
                direction = "-"
                time_spent = time.time() - c
                speed = (max_x - min_x) / time_spent
                rospy.loginfo("real speed: {} m/s".format(speed))
                flag = False
            elif pivot[0] < min_x:
                direction = "+"

            target_pose_2d = ut.get_body_pose(ut.get_body_id('cube_small_modified'))
            target_pose = ut.list_2_pose(target_pose_2d)
            pub.publish(target_pose)

            mico_moveit.add_box("cube", p.getBasePositionAndOrientation(cube), size=(0.05, 0.05, 0.05))
            mico_moveit.add_box("conveyor", p.getBasePositionAndOrientation(conveyor), size=(.1, .1, .02))
            time.sleep(.1)
    else:
        while True:
            target_pose_2d = ut.get_body_pose(ut.get_body_id('cube_small_modified'))
            target_pose = ut.list_2_pose(target_pose_2d)
            pub.publish(target_pose)
            mico_moveit.add_box("cube", p.getBasePositionAndOrientation(cube), size=(0.05, 0.05, 0.05))
            mico_moveit.add_box("conveyor", p.getBasePositionAndOrientation(conveyor), size=(.1, .1, .02))
            time.sleep(.1)


