import pybullet as p
import time
import pybullet_data
from mico_controller import MicoController
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

mico = p.loadURDF("/home/jxu/model.urdf", flags=p.URDF_USE_SELF_COLLISION)
mc = MicoController(mico)
mc.reset_arm_joint_values(mc.HOME)
cube = p.loadURDF("model/cube_small_modified.urdf", [0, -0.5, 0.025+0.01])
conveyor = p.loadURDF("model/conveyor.urdf", [0, -0.5, 0.01])

def generate_grasps(load_fnm=None, save_fnm=None, body="cube"):
    ## NOTE, now use sim-ann anf then switch

    if load_fnm:
        grasps = pickle.load(open(load_fnm, "rb"))
        return grasps.grasps
    else:
        gc = graspit_commander.GraspitCommander()
        gc.clearWorld()

        ## creat scene in graspit
        if body=="cube":
            floor_offset = -0.025  # half of the block size
            floor_pose = Pose(Point(-1, -1, floor_offset), Quaternion(0, 0, 0, 1))
            body_pose = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))
        elif body=="longBox":
            floor_offset = -0.09
            q = tft.quaternion_from_euler(0, 0.5*pi, 0)
            floor_pose = Pose(Point(-1, -1, floor_offset), Quaternion(0, 0, 0, 1))
            body_pose = Pose(Point(0, 0, 0), Quaternion(*q))
        else:
            raise ValueError("Invalid graspable body!")

        gc.importRobot('MicoGripper')
        gc.importGraspableBody(body, body_pose)
        gc.importObstacle('floor', floor_pose)
        grasps = gc.planGrasps()
        if save_fnm:
            pickle.dump(grasps, open(save_fnm, "wb"))
        return grasps.grasps

## Question: does ik need the pose to be in base_link

def get_world_grasps(grasps, objectID):
    """

    :param grasps: grasps.grasps returned by graspit
    :param objectID: object id
    :return: a list of tf tuples
    """
    object_pose = p.getBasePositionAndOrientation(objectID)
    world_T_object = tf_conversions.toMatrix(tf_conversions.fromTf(object_pose))
    grasps_in_world = list()
    for g in grasps:
        object_g = tf_conversions.toMatrix(tf_conversions.fromMsg(g.pose))
        world_g = world_T_object.dot(object_g)
        world_g_pose = tf_conversions.toMsg(tf_conversions.fromMatrix(world_g))
        # change end effector link
        old_ee_to_new_ee_translation_rotation = get_transfrom("m1n6s200_link_6", "m1n6s200_end_effector")
        world_g_pose_new = change_end_effector_link(world_g_pose, old_ee_to_new_ee_translation_rotation)
        grasps_in_world.append(tf_conversions.toTf(tf_conversions.fromMsg(world_g_pose_new)))
    return grasps_in_world

def pose_2_list(pose):
    """

    :param pose: geometry_msgs.msg.Pose
    :return: pose_2d: [[x, y, z], [x, y, z, w]]
    """
    position = [pose.position.x, pose.position.y, pose.position.z]
    orientation = [pose.quaternion.x, pose.quaternion.y, pose.quaternion.z, pose.quaternion.w]
    return [position, orientation]

def list_2_pose(pose_2d):
    """

    :param pose_2d: [[x, y, z], [x, y, z, w]]
    :return: pose: geometry_msgs.msg.Pose
    """
    return Pose(Point(*pose_2d[0]), Quaternion(*pose_2d[1]))


def display_grasp_pose_in_rviz(pose_2d_list, reference_frame):
    """

    :param pose_2d_list: a list of 2d array like poses
    :param reference_frame: which frame to reference
    """
    my_tf_manager = tf_manager.TFManager()
    for i, pose_2d in enumerate(pose_2d_list):
        pose = list_2_pose(pose_2d)
        ps = PoseStamped()
        ps.pose = pose
        ps.header.frame_id = reference_frame
        my_tf_manager.add_tf('G_{}'.format(i), ps)
        # import ipdb; ipdb.set_trace()
        my_tf_manager.broadcast_tfs()

# https://www.youtube.com/watch?v=aaDUIZVNCDM
def get_transfrom(reference_frame, target_frame):
    listener = tf.TransformListener()
    try:
        listener.waitForTransform(reference_frame, target_frame,
                                  rospy.Time(0), timeout=rospy.Duration(1))
        translation_rotation = listener.lookupTransform(reference_frame, target_frame,
                                                        rospy.Time())
    except Exception as e1:
        try:
            tf_buffer = tf2_ros.Buffer()
            tf2_listener = tf2_ros.TransformListener(tf_buffer)
            transform_stamped = tf_buffer.lookup_transform(reference_frame, target_frame,
                                                           rospy.Time(0), timeout=rospy.Duration(1))
            translation_rotation = tf_conversions.toTf(tf2_kdl.transform_to_kdl(transform_stamped))
        except Exception as e2:
            rospy.logerr("get_transfrom::\n " +
                         "Failed to find transform from %s to %s" % (
                             reference_frame, target_frame,))
    return translation_rotation

def change_end_effector_link(graspit_grasp_msg_pose, old_link_to_new_link_translation_rotation):
    """
    :param old_link_to_new_link_translation_rotation: geometry_msgs.msg.Pose,
        result of listener.lookupTransform((old_link, new_link, rospy.Time(0), timeout=rospy.Duration(1))
    :param graspit_grasp_msg_pose: The pose of a graspit grasp message i.e. g.pose
    ref_T_nl = ref_T_ol * ol_T_nl
    """
    graspit_grasp_pose_for_old_link_matrix = tf_conversions.toMatrix(
        tf_conversions.fromMsg(graspit_grasp_msg_pose)
    )

    old_link_to_new_link_tranform_matrix = tf.TransformerROS().fromTranslationRotation(
        old_link_to_new_link_translation_rotation[0],
        old_link_to_new_link_translation_rotation[1])
    graspit_grasp_pose_for_new_link_matrix = np.dot(graspit_grasp_pose_for_old_link_matrix,
                                                    old_link_to_new_link_tranform_matrix)  # ref_T_nl = ref_T_ol * ol_T_nl
    graspit_grasp_pose_for_new_link = tf_conversions.toMsg(
        tf_conversions.fromMatrix(graspit_grasp_pose_for_new_link_matrix))

    return graspit_grasp_pose_for_new_link

def back_off(pose_2d, offset):
    """
    Back up a grasp pose in world

    :param pose_2d: world pose, [[x, y, z], [x, y, z, w]]
    :param offset: the amount of distance to back off
    """
    world_T_old = tf_conversions.toMatrix(tf_conversions.fromTf(pose_2d))
    old_T_new = tf_conversions.toMatrix(tf_conversions.fromTf(((0, 0, -offset), (0, 0, 0, 1))))
    world_T_new = world_T_old.dot(old_T_new)
    pose_2d_new = tf_conversions.toTf(tf_conversions.fromMatrix(world_T_new))
    return  pose_2d_new

def visualize_grasp_in_graspit(pose_2d_list):
    pass
    # ## TODO
    # gc = graspit_commander.GraspitCommander()
    # gc.clearWorld()
    #
    # body = 'cube'
    # cube = ut.get_body_id("cube_small_modified")
    # body_pose = p.getBasePositionAndOrientation(cube)
    #
    # for pose_2d in pose_2d_list:
    #     new_g = tf_conversions.toMatrix(tf_conversions.fromTf(pose_2d))
    #     # change end effector link
    #     old_g = change_end_effector_link(world_g_pose, get_transfrom("m1n6s200_end_effector", "m1n6s200_link_6"))
    #     grasps_in_world.append(tf_conversions.toTf(tf_conversions.fromMsg(world_g_pose_new)))
    #
    # # floor_offset = -0.025  # half of the block size
    # # floor_pose = Pose(Point(-1, -1, floor_offset), Quaternion(0, 0, 0, 1))
    # body_pose = Pose(Point(*body_pose[0]), Quaternion(*body_pose[1]))
    #
    # gc.importRobot('MicoGripper')
    # gc.importGraspableBody(body, body_pose)
    # # gc.importObstacle('floor', floor_pose)
    #
    # pose_l = list()
    # for pose_2d in pose_2d_list:
    #     pose_l.append(list_2_pose(pose_2d))
    #
    # for pose in pose_l:
    #     gc.setRobotPose(pose)
    #     time.sleep(3)

if __name__ == "__main__":
    rospy.init_node("demo")

    ## starting pose
    mc.move_arm_joint_values(mc.HOME)
    mc.open_gripper()
    mc.mico_moveit.clear_scene()

    # mc.mico_moveit.add_box("cube", p.getBasePositionAndOrientation(cube), size=(0.048, 0.048, 0.18))
    mc.mico_moveit.add_box("cube", p.getBasePositionAndOrientation(cube), size=(0.05, 0.05, 0.05))
    mc.mico_moveit.add_box("conveyor", p.getBasePositionAndOrientation(conveyor), size=(.1, .1, .02))
    mc.mico_moveit.add_box("floor", ((0, 0, -0.005), (0, 0, 0, 1)), size=(2, 2, 0.01))

    # TODO 2. g_pose after backing off, is too far from object

    grasps = generate_grasps(load_fnm="grasps.pk", body="cube")
    grasps_in_world = get_world_grasps(grasps, cube)

    # g = back_off(grasps_in_world[0], 0.03)
    # mc.move_arm_eef_pose(g)
    # mc.close_gripper()
    # mc.move_arm_joint_values(mc.HOME)

    ##
    g_pose = None
    for i, g in enumerate(grasps_in_world):
        # the first grasp has ik if not including cube in the scene
        # backoff 0.01, 0.02, 0.005 does not help first grasp
        g = back_off(g, 0.01)
        j = mc.get_arm_ik(g)
        if j is None:
            print("no ik exists for the {}-th grasp".format(i))
        else:
            print("the {}-th grasp is reachable".format(i))
            g_pose = g
            break
    mc.move_arm_eef_pose(g_pose)


    # mc.reset_arm_joint_values(j_v)
    # print(mc.get_link_state(mc.ARM_EEF_INDEX))

    mc.close_gripper()
    mc.move_arm_joint_values(mc.HOME)



    while 1:
        time.sleep(1)
    # cubePos, cubeOrn = p.getBasePositionAndOrientation(mico)
    # print(cubePos,cubeOrn)
    # p.disconnect()


