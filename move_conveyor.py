import pybullet as p
import time
import math
import utils as ut


p.connect(p.SHARED_MEMORY)

if __name__ == "__main__":
    conveyor = ut.get_body_id("conveyor")
    ut.reset_body_base(conveyor, [[0, -0.3, 0.01], [0, 0, 0, 1]])

    cid = p.createConstraint(conveyor, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, -0.3, 0.01])
    a = 0
    direction = '+'
    try:
        while True:
            print(direction)
            print(a)
            if direction == "+":
                a = a + 0.005
            else:
                a = a - 0.005

            time.sleep(.1)
            pivot = [a, -0.3, 0.01]
            # p.changeConstraint(cid, pivot, jointChildFrameOrientation=orn, maxForce=50)
            p.changeConstraint(cid, pivot, maxForce=5000)
            if a > 0.3:
                direction = "-"
            elif a < -0.3:
                direction = "+"
    except KeyboardInterrupt:
        print("interrupt")
        ut.remove_all_constraints()
