import pybullet as p
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
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
from math import pi
import tf.transformations as tft
import utils as ut
import time
import rospkg
import os
import motion_prediction.srv
import csv
import argparse
from reachability_utils.process_reachability_data_from_csv import load_reachability_data_from_dir
from reachability_utils.reachability_resolution_analysis import interpolate_pose_in_reachability_space_grid


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('-o', '--object_name', type=str, default= 'cube',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('-v', '--conveyor_velocity', type=float, default=0.03,
                        help='Velocity of conveyor belt')
    parser.add_argument('-d', '--conveyor_distance', type=float, default=0.5,
                        help="Distance of conveyor belt to robot base")
    args = parser.parse_args()

    args.mesh_dir = os.path.abspath('meshes/meshes_obj')
    args.object_mesh_filepath = os.path.join(args.mesh_dir , '{}.obj'.format(args.object_name))

    # set ROS parameters
    rospy.set_param('conveyor_velocity', args.conveyor_velocity)
    rospy.set_param('object_name', args.object_name)
    rospy.set_param('object_mesh_filepath', args.object_mesh_filepath)

    # set_up urdf
    urdf_template_filepath = 'model/object_template.urdf'
    urdf_target_object_filepath = 'model/target_object.urdf'
    os.system('cp {} {}'.format(urdf_template_filepath, urdf_target_object_filepath))
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name.obj', args.object_mesh_filepath, urdf_target_object_filepath)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name', args.object_name, urdf_target_object_filepath)
    os.system(sed_cmd)
    args.urdf_target_object_filepath = urdf_target_object_filepath

    args.reachability_data_dir = os.path.join(rospkg.RosPack().get_path('mico_reachability_config'), 'data')
    args.step_size, args.mins, args.dims = load_reachability_params(args.reachability_data_dir)

    return args


def load_reachability_params(reachability_data_dir):
    step_size = np.loadtxt(os.path.join(reachability_data_dir, 'reach_data.step'), delimiter=',')
    mins = np.loadtxt(os.path.join(reachability_data_dir, 'reach_data.mins'), delimiter=',')
    dims = np.loadtxt(os.path.join(reachability_data_dir, 'reach_data.dims'), delimiter=',', dtype=int)

    return step_size, mins, dims

## TODO uniform sampling grasps

def generate_grasps(load_fnm=None, save_fnm=None, body="cube"):
    """
    This method assumes the target object to be at world origin.

    :param load_fnm: load file name
    :param save_fnm: save file name
    :param body: the name of the graspable object to load in graspit
    :return: graspit grasps.grasps
    """
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

def plan_reachable_grasps(load_fnm=None, save_fnm=None, object_name="cube", object_pose_2d=None, seed_grasp=None, max_steps=35000):
    """ return world grasps in pose_2d """
    if load_fnm:
        grasps = pickle.load(open(load_fnm, "rb"))
        return grasps
    else:
        gc = graspit_commander.GraspitCommander()
        gc.clearWorld()

    gc.importRobot('MicoGripper')
    gc.setRobotPose(Pose(Point(0, 0, 1), Quaternion(0, 0, 0, 1)))
    gc.importGraspableBody(object_name, ut.list_2_pose(object_pose_2d))
    # hard-code floor constraint
    gc.importObstacle('floor', Pose(Point(object_pose_2d[0][0]-2, object_pose_2d[0][1]-2, 0), Quaternion(0, 0, 0, 1)))
    # TODO not considering conveyor?

    # simulated annealling
    grasps = gc.planGrasps(max_steps=max_steps+35000, search_energy='REACHABLE_FIRST_HYBRID_GRASP_ENERGY',
                           use_seed_grasp=seed_grasp is not None, seed_grasp=seed_grasp)
    grasps = grasps.grasps

    # keep only good grasps
    # TODO is this really required?
    good_grasps = [g for g in grasps if g.volume_quality > 0]

    # for g in good_grasps:
    #     gc.setRobotPose(g.pose)
    #     time.sleep(3)

    # change end effector link
    grasps_in_world = list()
    for g in good_grasps:
        old_ee_to_new_ee_translation_rotation = get_transfrom("m1n6s200_link_6", "m1n6s200_end_effector")
        g_pose_new = change_end_effector_link(g.pose, old_ee_to_new_ee_translation_rotation)
        grasps_in_world.append(tf_conversions.toTf(tf_conversions.fromMsg(g_pose_new)))

    if save_fnm:
        pickle.dump(grasps_in_world, open(save_fnm, "wb"))
    return grasps_in_world

def get_world_grasps(grasps, objectID, object_pose=None):
    """

    :param grasps: grasps.grasps returned by graspit
    :param objectID: object id
    :param object_pose: the pose of the target; if None, use current pose
    :return: a list of tf tuples
    """
    if object_pose is None:
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

def display_grasp_pose_in_rviz(pose_2d_list, reference_frame):
    """

    :param pose_2d_list: a list of 2d array like poses
    :param reference_frame: which frame to reference
    """
    my_tf_manager = tf_manager.TFManager()
    for i, pose_2d in enumerate(pose_2d_list):
        pose = ut.list_2_pose(pose_2d)
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

def step_simulate(t):
    """ using p.stepSimulation with p.setTimeStep a large time (like 1s) is unpredictable"""
    n = int(round(t*240))
    for i in range(n):
        p.stepSimulation()
        time.sleep(1.0/240.0)

def predict(duration, body):
    """
    assume we know the direction and speed, predict linear motion
    :param body: target body id
    """
    speed = 0.03
    direction = '+'
    pose = ut.get_body_pose(body)
    pose[0][0] = pose[0][0] + speed * duration
    return pose

def can_grasp(g_position, d_gpos_threshold=None, d_target_threshold=None):
    """

    :param g_position: position of the grasp that we want to match, normally just use pre-grasp
    :param d_gpos_threshold: allowed position distance between eef and grasp position
    :param d_target_threshold: allowed position distance between eef and target
    :return:
    """
    # TODO shall I give constraint on quaternion as well?
    target_position = ut.get_body_pose(target_object)[0]
    eef_position = mc.get_arm_eef_pose()[0]
    d_target = np.linalg.norm(np.array(target_position) - np.array(eef_position))
    d_gpos = np.linalg.norm(np.array(g_position) - np.array(eef_position))
    rospy.loginfo("distance to target: {}".format(d_target))
    rospy.loginfo("distance to g_position: {}".format(d_gpos))

    if d_gpos_threshold is not None and d_target_threshold is None:
        can = d_gpos < d_gpos_threshold
    elif d_gpos_threshold is None and d_target_threshold is not None:
        can = d_target < d_target_threshold
    elif d_gpos_threshold is not None and d_target_threshold is not None:
        can = d_gpos < d_gpos_threshold and d_target < d_target_threshold
    else:
        raise ValueError("please specify a threshold!")
    return can

def is_success(target_object):
    target_pose_2d = ut.get_body_pose(target_object)
    if target_pose_2d[0][2] > 0.05:
        return True
    else:
        return False

def get_reachability_of_grasps_pose(grasps_in_world, sdf_reachability_space):
    """ grasps_in_world is a list of geometry_msgs/Pose """
    sdf_values = []
    for g_pose in grasps_in_world:
        trans, rot = tf_conversions.toTf(tf_conversions.fromMsg(g_pose))
        rpy = tf_conversions.Rotation.Quaternion(*rot).GetRPY()
        query_pose = np.concatenate((trans, rpy))
        sdf_values.append(
            interpolate_pose_in_reachability_space_grid(sdf_reachability_space,
                                                        args.mins, args.step_size, args.dims, query_pose))

    # is_reachable = [sdf_values[i] > 0 for i in range(len(sdf_values))]
    return sdf_values

def get_reachability_of_grasps_pose_2d(grasps_in_world, sdf_reachability_space):
    """ grasps_in_world is a list of pose_2d """
    sdf_values = []
    for g_pose in grasps_in_world:
        trans, rot = g_pose[0], g_pose[1]
        rpy = tf_conversions.Rotation.Quaternion(*rot).GetRPY()
        query_pose = np.concatenate((trans, rpy))
        sdf_values.append(
            interpolate_pose_in_reachability_space_grid(sdf_reachability_space,
                                                        args.mins, args.step_size, args.dims, query_pose))

    # is_reachable = [sdf_values[i] > 0 for i in range(len(sdf_values))]
    return sdf_values


if __name__ == "__main__":
    args = get_args()
    _, mins, step_size, dims, sdf_reachability_space = load_reachability_data_from_dir(args.reachability_data_dir)

    rospy.init_node("demo")

    physicsClient = p.connect(p.GUI_SERVER)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    ut.reset_camera(yaw=-24.400014877319336, pitch=-47.000030517578125, dist=0.9, target=(-0.2, -0.30000001192092896, 0.0))
    ut.remove_all_bodies()
    # p.setAdditionalSearchPath("/home/jxu/bullet3/data") #optionally
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    # /home/jxu/.local/lib/python2.7/site-packages/pybullet_data
    # /home/jxu/bullet3/examples/pybullet/examples

    ONLY_TRACKING = False
    DYNAMIC = True
    KEEP_PREVIOUS_GRASP = True
    RANK_BY_REACHABILITY = True
    LOAD_OBSTACLES = False
    ONLINE_PLANNING = True

    p.setGravity(0, 0, -9.8)

    ## load plane, conveyor
    plane = p.loadURDF("plane.urdf")
    target_object = p.loadURDF(args.urdf_target_object_filepath, [-0.8, -args.conveyor_distance, 0.025 + 0.01])
    conveyor = p.loadURDF("model/conveyor.urdf", [-0.8, -args.conveyor_distance, 0.01])

    ## draw lines
    p.addUserDebugLine(lineFromXYZ=[-0.8-0.05, -args.conveyor_distance+0.05, 0], lineToXYZ=[0.8+0.05, -args.conveyor_distance+0.05, 0], lineWidth=5, lifeTime=0, lineColorRGB=[0, 0, 0])
    p.addUserDebugLine(lineFromXYZ=[-0.8-0.05, -args.conveyor_distance-0.05, 0], lineToXYZ=[0.8+0.05, -args.conveyor_distance-0.05, 0], lineWidth=5, lifeTime=0, lineColorRGB=[0, 0, 0])

    ## load robot
    urdf_dir = os.path.join(rospkg.RosPack().get_path('kinova_description'), 'urdf')
    urdf_filename = 'mico.urdf'
    if not os.path.exists(os.path.join(urdf_dir, urdf_filename)):
        os.system('cp model/mico.urdf {}'.format(os.path.join(urdf_dir, urdf_filename)))
    mico = p.loadURDF(os.path.join(urdf_dir, urdf_filename), flags=p.URDF_USE_SELF_COLLISION)

    ## load obstacles
    if LOAD_OBSTACLES:
        all_bottle_obstacle = p.loadURDF("model/all_bottle_obstacle.urdf", [0.08998222825053534, -0.3499910643580809+0.5-args.conveyor_distance, 0.00023844586980673307], [-0.0216622233067531,-0.0056517363083496315,0.9252930239049401,0.3785916347081129])
        trash_can_new_obstacle = p.loadURDF("model/trash_can_new_obstacle.urdf", [-0.30547354355304757, -0.30201510506407675+0.5-args.conveyor_distance, 0.06916064731741166],[0.014904781746875194,-0.039341362803788846,0.0011919812701876302,0.9991139493743795])
        gillette_shaving_gel_obstacle = p.loadURDF("model/gillette_shaving_gel_obstacle.urdf",[-0.09985463045431538, -0.5998251882677045+0.5-args.conveyor_distance, 0.000567468746735579],[-0.035094655962893954,0.011424311280120345,-0.003006735972672182,0.9993141697051092])

    ## memory leaks happen sometimes without this but a breakpoint
    p.setRealTimeSimulation(1)

    ## initialize controller
    mc = MicoController(mico)
    mc.reset_arm_joint_values(mc.HOME)

    ## starting pose
    mc.move_arm_joint_values(mc.HOME, plan=False)
    mc.open_gripper()
    mc.mico_moveit.clear_scene()

    mc.mico_moveit.add_mesh(args.object_name, ut.get_body_pose(target_object), args.object_mesh_filepath)
    mc.mico_moveit.add_box("conveyor", ut.get_body_pose(conveyor), size=(.1, .1, .02))
    mc.mico_moveit.add_box("floor", ((0, 0, -0.005), (0, 0, 0, 1)), size=(2, 2, 0.01))
    if LOAD_OBSTACLES:
        mc.mico_moveit.add_mesh("all_bottle_obstacle", ut.get_body_pose(all_bottle_obstacle), 'meshes/meshes_obj/all_bottle.obj')
        mc.mico_moveit.add_mesh("trash_can_new_obstacle", ut.get_body_pose(trash_can_new_obstacle), 'meshes/meshes_obj/trash_can_new.obj')
        mc.mico_moveit.add_mesh("gillette_shaving_gel_obstacle", ut.get_body_pose(gillette_shaving_gel_obstacle), 'meshes/meshes_obj/gillette_shaving_gel.obj')

    # kalman filter service
    rospy.wait_for_service('motion_prediction')
    motion_predict_svr = rospy.ServiceProxy('motion_prediction', motion_prediction.srv.GetFuturePose)

    print("here")
    pre_position_trajectory = None
    if ONLINE_PLANNING:
        object_pose_2d = ut.get_body_pose(target_object)
        # grasps_in_world = plan_reachable_grasps(save_fnm='grasps_online.pk', object_name='cube',
        #                                         object_pose_2d=object_pose_2d, max_steps=10000)
        grasps_in_world = plan_reachable_grasps(load_fnm='grasps_online.pk')
    else:
        grasps = generate_grasps(load_fnm="grasps.pk", body="cube")

    start_time = time.time()
    video_fname = '{}-{}.mp4'.format(args.object_name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join("video",video_fname))

    pre_g_index = None
    while True:
        #### grasp planning
        c = time.time()
        current_pose = ut.get_body_pose(target_object)
        if current_pose[0][0] > 0.79:
            # target moves outside workspace, break directly
            break
        future_pose = [list(motion_predict_svr(duration=1).prediction), current_pose[1]]
        future_pose_p = predict(1, target_object)
        grasps_in_world = get_world_grasps(grasps, target_object, future_pose)

        pre_grasps_in_world = list()
        for g in grasps_in_world:
            pre_grasps_in_world.append(MicoController.back_off(g, 0.05))

        pre_g_pose = None
        g_pose = None
        pre_g_joint_values = None
        g_index = None

        # TODO, now just check reachability of pregrasp with target
        # pre_grasps_in_world is always updated with the current pose of the target object
        if RANK_BY_REACHABILITY:
            if KEEP_PREVIOUS_GRASP and pre_g_index is not None and \
                    get_reachability_of_grasps_pose_2d([pre_grasps_in_world[pre_g_index]], sdf_reachability_space)[0] > 0:
                rospy.loginfo("the previous pre-grasp is reachable")
                pre_g_pose = pre_grasps_in_world[pre_g_index]
                g_pose = grasps_in_world[pre_g_index]
                pre_g_joint_values = mc.get_arm_ik(pre_grasps_in_world[pre_g_index])
                g_index = pre_g_index
                if pre_g_joint_values is None:
                    rospy.logerr("the pre-grasp pose is actually not reachable")
            else:
                sdf_values = get_reachability_of_grasps_pose_2d(pre_grasps_in_world, sdf_reachability_space)
                print(max(sdf_values))
                if max(sdf_values) > 0:
                    g_index = int(np.argmax(sdf_values))
                    pre_g_pose = pre_grasps_in_world[g_index]
                    g_pose = grasps_in_world[g_index]
                    pre_g_joint_values = mc.get_arm_ik(pre_grasps_in_world[g_index])
                    pre_g_index = g_index
                    if pre_g_joint_values is None:
                        rospy.logerr("the pre-grasp pose is actually not reachable")
        else:
            if KEEP_PREVIOUS_GRASP and pre_g_index is not None and mc.get_arm_ik(pre_grasps_in_world[pre_g_index]) is not None:
                # always first check whether the previous grasp is still reachable
                rospy.loginfo("the previous pre-grasp is reachable")
                pre_g_pose = pre_grasps_in_world[pre_g_index]
                g_pose = grasps_in_world[pre_g_index]
                pre_g_joint_values = mc.get_arm_ik(pre_grasps_in_world[pre_g_index])
                g_index = pre_g_index
            else:
                # go through the list ranked by stability
                for i, g in enumerate(pre_grasps_in_world):
                    tt = time.time()
                    j = mc.get_arm_ik(g)
                    print("get arm ik takes {}".format(time.time()-tt))
                    if j is None:
                        pass
                        # print("no ik exists for the {}-th pre-grasp".format(i))
                    else:
                        rospy.loginfo("the {}-th pre-grasp is reachable".format(i))
                        pre_g_pose = g
                        g_pose = grasps_in_world[i]
                        pre_g_joint_values = j
                        g_index = i
                        pre_g_index = g_index
                        break

        # did not find a reachable pre-grasp
        if pre_g_pose is None:
            rospy.loginfo("object out of range!")
            continue
        rospy.loginfo("grasp planning takes {}".format(time.time()-c))

        #### move to pre-grasp pose
        looking_ahead = 3
        rospy.loginfo("trying to reach pre-grasp pose {}".format(pre_g_pose))
        c = time.time()
        rospy.loginfo("previous trajectory is reaching: {}".format(mc.seq))

        if pre_position_trajectory is None:
            position_trajectory = mc.plan_arm_joint_values(goal_joint_values=pre_g_joint_values)
        else:
            start_index = min(mc.seq+looking_ahead, len(pre_position_trajectory)-1)
            position_trajectory = mc.plan_arm_joint_values(goal_joint_values=pre_g_joint_values, start_joint_values=pre_position_trajectory[start_index])
        rospy.loginfo("planning takes {}".format(time.time()-c))

        if position_trajectory is None:
            rospy.loginfo("No plans found!")
        else:
            rospy.loginfo("start executing")
            pre_position_trajectory = position_trajectory # just another reference
            mc.execute_arm_trajectory(position_trajectory)
            time.sleep(0.2)

        # TODO sometimes grasp planning takes LONGER with some errors after tracking for a long time, This results the previous
        # trajectory to have finished before we send another goal to move arm
        # TODO add blocking to control

        #### grasp
        if ONLY_TRACKING:
            pass
        else:
            # if can_grasp(pre_g_pose[0], 0.05, None):
            if can_grasp(pre_g_pose[0], None, 0.055):
                if DYNAMIC:
                    rospy.loginfo("start grasping")
                    # predicted_pose = predict(1, target_object)
                    predicted_pose = [list(motion_predict_svr(duration=1).prediction), current_pose[1]]
                    g_pose = get_world_grasps([grasps[g_index]], target_object, predicted_pose)[0]
                    mc.move_arm_eef_pose(g_pose, plan=False) # TODO sometimes this motion is werid? rarely
                    # time.sleep(1) # give sometime to move before closing
                    mc.close_gripper()
                    mc.cartesian_control(z=0.05)
                    # NOTE: The trajectory returned by this will have a far away first waypoint to jump to
                    # and I assume it is because the initial position is not interpreted as valid by moveit
                    # or the good first waypoint is blocked by a instantly updated block scene
                    # mc.move_arm_joint_values(mc.HOME)
                    break
                else:
                    mc.grasp(pre_g_pose, DYNAMIC)
                    break

    rospy.sleep(1) # give some time for lift to finish before get time
    time_spent = time.time() - start_time
    rospy.loginfo("time spent: {}".format(time_spent))
    rospy.sleep(2)

    # check success and then do something
    success = is_success(target_object)
    rospy.loginfo(success)

    # save result to file: object_name, success, time, velocity, conveyor_distance
    result_dir = 'results'
    result = {'object_name': args.object_name,
              'success': success,
              'time': time_spent,
              'video_filename': video_fname,
              'conveyor_velocity': args.conveyor_velocity,
              'conveyor_distance': args.conveyor_distance}

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file_path = os.path.join(result_dir, '{}.csv'.format(args.object_name))
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


    p.stopStateLogging(logging)
    # kill all other things
    # os.system("kill -9 $(pgrep -f move_conveyor)")
    os.system("kill -9 $(pgrep -f trajectory_execution_server)")
    os.system("kill -9 $(pgrep -f motion_prediction_server)")

    while 1:
        time.sleep(1)
    # p.disconnect()