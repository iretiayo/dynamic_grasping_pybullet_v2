from __future__ import division
import pybullet as p
import pybullet_data
from mico_controller import MicoController
import rospy
import tf_conversions
import numpy as np
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)
import utils as ut
import time
import rospkg
import os
import motion_prediction.srv
import csv
import argparse
from reachability_utils.process_reachability_data_from_csv import load_reachability_data_from_dir
import skfmm
import trimesh
import yaml

from grasp_utils import load_reachability_params, get_transfrom, convert_grasps, change_end_effector_link, \
    plan_reachable_grasps, get_world_grasps, get_reachability_of_grasps_pose, generate_grasps, \
    create_occupancy_grid_from_obstacles


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('-o', '--object_name', type=str, default= 'cube',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('-v', '--conveyor_velocity', type=float, default=0.03,
                        help='Velocity of conveyor belt')
    parser.add_argument('-d', '--conveyor_distance', type=float, default=0.5,
                        help="Distance of conveyor belt to robot base")
    args = parser.parse_args()

    args.video_dir = 'videos'
    if not os.path.exists(args.video_dir):
        os.makedirs(args.video_dir)

    args.result_dir = 'results'
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    args.mesh_dir = os.path.abspath('model')
    args.object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name),
                                             '{}.obj'.format(args.object_name))
    args.object_mesh_filepath_ply = os.path.join(args.mesh_dir, '{}'.format(args.object_name),
                                             '{}.ply'.format(args.object_name))

    target_mesh = trimesh.load_mesh(args.object_mesh_filepath)
    args.target_extents = target_mesh.bounding_box.extents.tolist()

    # set ROS parameters
    rospy.set_param('conveyor_velocity', args.conveyor_velocity)
    rospy.set_param('object_name', args.object_name)
    rospy.set_param('object_mesh_filepath', args.object_mesh_filepath)
    rospy.set_param('target_extents', args.target_extents)
    if args.conveyor_velocity == 0.01:
        conveyor_extent = [-0.6, 0.6]
    else:
        conveyor_extent = [-0.8, 0.8]
    rospy.set_param('conveyor_extent', conveyor_extent)
    args.min_x, args.max_x = conveyor_extent

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

    args.ONLY_TRACKING = False
    args.DYNAMIC = True
    args.KEEP_PREVIOUS_GRASP = True
    args.RANK_BY_REACHABILITY = True
    args.LOAD_OBSTACLES = True
    args.ONLINE_PLANNING = False

    config = yaml.load(open('scene.yaml'))
    args.__dict__['scene_config'] = config

    return args


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


def is_success(target_object, original_height):
    target_pose_2d = ut.get_body_pose(target_object)
    if target_pose_2d[0][2] > original_height + 0.02:
        return True
    else:
        return False


def create_scence(args):

    p.connect(p.GUI_SERVER)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetSimulation()
    p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=-24.4, cameraPitch=-47.0, cameraTargetPosition=(-0.2, -0.30, 0.0))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # load plane, conveyor and target
    p.loadURDF("plane.urdf")
    target_object_id = p.loadURDF(args.urdf_target_object_filepath, [args.min_x, -args.conveyor_distance, args.target_extents[2]/2 + 0.01])
    conveyor_id = p.loadURDF("model/conveyor.urdf", [args.min_x, -args.conveyor_distance, 0.01])

    # draw lines
    p.addUserDebugLine(lineFromXYZ=[args.min_x-0.05, -args.conveyor_distance+0.05, 0], lineToXYZ=[args.max_x+0.05, -args.conveyor_distance+0.05, 0], lineWidth=5, lifeTime=0, lineColorRGB=[0, 0, 0])
    p.addUserDebugLine(lineFromXYZ=[args.min_x-0.05, -args.conveyor_distance-0.05, 0], lineToXYZ=[args.max_x+0.05, -args.conveyor_distance-0.05, 0], lineWidth=5, lifeTime=0, lineColorRGB=[0, 0, 0])

    # load robot
    urdf_dir = os.path.join(rospkg.RosPack().get_path('kinova_description'), 'urdf')
    urdf_filename = 'mico.urdf'
    if not os.path.exists(os.path.join(urdf_dir, urdf_filename)):
        os.system('cp model/mico.urdf {}'.format(os.path.join(urdf_dir, urdf_filename)))
    mico_id = p.loadURDF(os.path.join(urdf_dir, urdf_filename), flags=p.URDF_USE_SELF_COLLISION)

    # load obstacles
    if args.LOAD_OBSTACLES:
        for obstacle in args.scene_config['obstacles']:
            obs_pos = obstacle['position']
            obs_pos[1] += 0.5-args.conveyor_distance
            obs_id = p.loadURDF(obstacle['urdf_filepath'], obs_pos, obstacle['orientation'])
            obstacle['body_id'] = obs_id

    # memory leaks happen sometimes without this but a breakpoint
    p.setRealTimeSimulation(1)
    rospy.sleep(2)  # time for objects to stabilize

    args.mico_id, args.conveyor_id, args.target_object_id = mico_id, conveyor_id, target_object_id
    rospy.set_param('target_object_id', args.target_object_id)
    rospy.set_param('conveyor_id', args.conveyor_id)


def create_scence_moveit(args, mc):
    mc.mico_moveit.clear_scene()

    mc.mico_moveit.add_mesh(args.object_name, ut.get_body_pose(args.target_object_id), args.object_mesh_filepath)
    mc.mico_moveit.add_box("conveyor", ut.get_body_pose(args.conveyor_id), size=(.1, .1, .02))
    mc.mico_moveit.add_box("floor", ((0, 0, -0.005), (0, 0, 0, 1)), size=(2, 2, 0.01))

    if args.LOAD_OBSTACLES:
        for obstacle in args.scene_config['obstacles']:
            mc.mico_moveit.add_mesh(obstacle['object_name'], ut.get_body_pose(obstacle['body_id']), obstacle['obj_filepath'])
        time.sleep(1)


def get_reachability_space(args):
    rospy.loginfo("start creating sdf reachability space...")
    c = time.time()
    if args.LOAD_OBSTACLES:
        obstacle_mesh_filepaths, obstacle_poses = [], []
        for obstacle in args.scene_config['obstacles']:
            obstacle_mesh_filepaths.append(obstacle['ply_filepath'])
            obstacle_poses.append(ut.list_2_pose(ut.get_body_pose(obstacle['body_id'])))

        # create reachability space from obstacles
        obstacles_mask_3d = create_occupancy_grid_from_obstacles(obstacle_mesh_filepaths=obstacle_mesh_filepaths,
                                                                 obstacle_poses=obstacle_poses,
                                                                 mins_xyz=args.mins[:3],
                                                                 step_size_xyz=args.step_size[:3],
                                                                 dims_xyz=args.dims[:3])
        binary_reachability_space, mins, step_size, dims, _ = load_reachability_data_from_dir(
            args.reachability_data_dir)
        # embed obstacles into reachability space
        binary_reachability_space[obstacles_mask_3d > 0] = 0

        # Generate sdf
        binary_reachability_space -= 0.5
        sdf_reachability_space = skfmm.distance(binary_reachability_space, periodic=[False, False, False, True, True, True])
        binary_reachability_space += 0.5  # undo previous operation
        # sdf_reachability_space.tofile(open(os.path.join(args.reachability_data_dir, 'reach_data') + '.sdf', 'w'))
    else:
        _, mins, step_size, dims, sdf_reachability_space = load_reachability_data_from_dir(args.reachability_data_dir)
    rospy.loginfo("sdf reachability space created, which takes {}".format(time.time()-c))
    return sdf_reachability_space


if __name__ == "__main__":
    args = get_args()
    create_scence(args)

    rospy.init_node("demo")

    # initialize controller
    mc = MicoController(args.mico_id)
    mc.reset_arm_joint_values(mc.HOME)

    # starting pose
    mc.move_arm_joint_values(mc.HOME, plan=False)
    mc.open_gripper()

    create_scence_moveit(args, mc)
    rospy.set_param("start_moving", False)

    # load grasps
    starting_line = -0.6
    if args.ONLINE_PLANNING:
        # need to make sure seed grasp is good!
        grasp_fnm = "grasps_online_"+str(args.conveyor_distance)+'.pk'
        pose_init = [[starting_line, args.conveyor_distance, ut.get_body_pose(args.target_object_id)[0][2]], [0, 0, 0, 1]]
        # grasps_in_world = plan_reachable_grasps(save_fnm=grasp_fnm, object_name='cube',
        #                                         object_pose_2d=pose_init, max_steps=10000)
        grasps_in_world = plan_reachable_grasps(load_fnm=grasp_fnm)
    else:
        grasp_filepath = os.path.join(args.mesh_dir, args.object_name, "grasps_"+str(args.object_name)+'.pk')
        if os.path.exists(grasp_filepath):
            grasps = generate_grasps(load_fnm=grasp_filepath, body=args.object_mesh_filepath_ply, body_extents=args.target_extents)
        else:
            grasps = generate_grasps(save_fnm=grasp_filepath, body=args.object_mesh_filepath_ply, body_extents=args.target_extents)
        grasp_poses_2d_before_eef_transformation = [ut.pose_2_list(g.pose) for g in grasps]

    # create sdf reachability space
    if args.RANK_BY_REACHABILITY:
        sdf_reachability_space = get_reachability_space(args)
    rospy.set_param("start_moving", True)

    # kalman filter service
    rospy.wait_for_service('motion_prediction')
    motion_predict_svr = rospy.ServiceProxy('motion_prediction', motion_prediction.srv.GetFuturePose)

    print("here")
    pre_position_trajectory = None
    start_time = None

    video_fname = '{}-{}.mp4'.format(args.object_name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join(args.video_dir, video_fname))


    #############################################################################################
    previous_g_index = None # previous grasp pose index
    original_height = ut.get_body_pose(args.target_object_id)[0][2]
    old_ee_to_new_ee_translation_rotation = get_transfrom("m1n6s200_link_6", "m1n6s200_end_effector")
    new_ee_to_old_ee_translation_rotation = get_transfrom("m1n6s200_end_effector", "m1n6s200_link_6")

    if args.ONLINE_PLANNING:
        grasp_old = grasps_in_world[0]
        pose_old = pose_init

    while True:
        current_pose = ut.get_body_pose(args.target_object_id)
        current_conveyor_pose = ut.get_body_pose(args.conveyor_id)

        if start_time is None:
            # enter workspace
            if current_pose[0][0] > starting_line:
                start_time = time.time()
            continue

        if current_conveyor_pose[0][0] > args.max_x-0.05:
            # target moves outside workspace, break directly
            break

        #### grasp planning
        c = time.time()
        future_pose = tf_conversions.toTf(tf_conversions.fromMsg(motion_predict_svr(duration=1).prediction.pose))

        # information about the current grasp
        pre_g_pose = None
        g_pose = None
        pre_g_joint_values = None
        g_index = None

        if args.ONLINE_PLANNING:
            grasp_old = convert_grasps([grasp_old], pose_old, future_pose)[0]
            pose_old = future_pose
            pre_g_pose = MicoController.back_off(grasp_old, 0.05)
            j = mc.get_arm_ik(pre_g_pose)
            if j is not None: # previous grasp after conversion is still reachabale
                rospy.loginfo("the previous pre-grasp is still reachabale")
                g_pose = grasp_old
                pre_g_joint_values = j
            else:
                from graspit_interface.msg import Grasp
                seed_grasp = Grasp()
                seed_grasp.pose = change_end_effector_link(ut.list_2_pose(grasp_old),
                                                           new_ee_to_old_ee_translation_rotation)
                temp_list = plan_reachable_grasps(object_name='cube', object_pose_2d=future_pose,
                                                      seed_grasp=seed_grasp, max_steps=10)
                if len(temp_list) == 0:
                    # this is now not possible
                    rospy.loginfo("online planning does not find a good grasp, just continue..")
                    continue
                g_pose = temp_list[0]
                pre_g_pose = MicoController.back_off(g_pose, 0.05)
                j = mc.get_arm_ik(pre_g_pose)
                if j is None:
                    rospy.logerr("the online planned pre-grasp is actually not reachable")
                    continue
                else:
                    pre_g_joint_values = j
                    rospy.loginfo("got an online planned pre-grasp")

        else:
            grasps_in_world, grasps_in_world_before_eef_trans = get_world_grasps(grasps, args.target_object_id, old_ee_to_new_ee_translation_rotation, future_pose)
            pre_grasps_in_world = list()
            pre_grasps_in_world_before_eef_trans = list()
            for g in grasps_in_world:
                pre_grasps_in_world.append(MicoController.back_off(g, 0.05))
            for g in grasps_in_world_before_eef_trans:
                pre_grasps_in_world_before_eef_trans.append(MicoController.back_off(g, 0.05))

            #################################################################################################
            ####################### grasp switching with reachability space #################################
            if args.RANK_BY_REACHABILITY:
                if args.KEEP_PREVIOUS_GRASP and previous_g_index is not None:
                    j = mc.get_arm_ik(pre_grasps_in_world[previous_g_index])
                    if j is not None:
                        rospy.loginfo("the previous pre-grasp is reachable")
                        pre_g_pose = pre_grasps_in_world[previous_g_index]
                        g_pose = grasps_in_world[previous_g_index]
                        pre_g_joint_values = j
                        g_index = previous_g_index
                    else:
                        pre_grasp_poses_before_eef_transformation = [ut.list_2_pose(g) for g in
                                                                     pre_grasps_in_world_before_eef_trans]
                        sdf_values = get_reachability_of_grasps_pose(pre_grasp_poses_before_eef_transformation,
                                                                     sdf_reachability_space, args.mins, args.step_size, args.dims)
                        g_index = int(np.argmax(sdf_values))
                        pre_g_pose = pre_grasps_in_world[g_index]
                        g_pose = grasps_in_world[g_index]
                        pre_g_joint_values = mc.get_arm_ik(pre_grasps_in_world[g_index])
                        previous_g_index = g_index
                        print(g_index)
                        print(sdf_values)
                        if pre_g_joint_values is None:
                            rospy.logerr("the pre-grasp pose is actually not reachable")
                            continue
                else:
                    pre_grasp_poses_before_eef_transformation = [ut.list_2_pose(g) for g in
                                                                 pre_grasps_in_world_before_eef_trans]
                    sdf_values = get_reachability_of_grasps_pose(pre_grasp_poses_before_eef_transformation,
                                                                 sdf_reachability_space, args.mins, args.step_size, args.dims)
                    g_index = int(np.argmax(sdf_values))
                    pre_g_pose = pre_grasps_in_world[g_index]
                    g_pose = grasps_in_world[g_index]
                    pre_g_joint_values = mc.get_arm_ik(pre_grasps_in_world[g_index])
                    previous_g_index = g_index
                    print(g_index)
                    print(sdf_values)
                    if pre_g_joint_values is None:
                        rospy.logerr("the pre-grasp pose is actually not reachable")
                        continue

            #################################################################################################
            ####################### grasp switching without reachability space ##############################
            else:
                if args.KEEP_PREVIOUS_GRASP and previous_g_index is not None:
                    j = mc.get_arm_ik(pre_grasps_in_world[previous_g_index])
                    if j is not None:
                    # always first check whether the previous grasp is still reachable
                        rospy.loginfo("the previous pre-grasp is reachable")
                        pre_g_pose = pre_grasps_in_world[previous_g_index]
                        g_pose = grasps_in_world[previous_g_index]
                        pre_g_joint_values = j
                        g_index = previous_g_index
                    else:
                        # go through the list ranked by stability
                        for i, g in enumerate(pre_grasps_in_world):
                            # tt = time.time()
                            j = mc.get_arm_ik(g)
                            # print("get arm ik takes {}".format(time.time()-tt))
                            if j is None:
                                pass
                                # print("no ik exists for the {}-th pre-grasp".format(i))
                            else:
                                rospy.loginfo("the {}-th pre-grasp is reachable".format(i))
                                pre_g_pose = g
                                g_pose = grasps_in_world[i]
                                pre_g_joint_values = j
                                g_index = i
                                previous_g_index = g_index
                                break
                else:
                    # go through the list ranked by stability
                    for i, g in enumerate(pre_grasps_in_world):
                        # tt = time.time()
                        j = mc.get_arm_ik(g)
                        # print("get arm ik takes {}".format(time.time()-tt))
                        if j is None:
                            pass
                            # print("no ik exists for the {}-th pre-grasp".format(i))
                        else:
                            rospy.loginfo("the {}-th pre-grasp is reachable".format(i))
                            pre_g_pose = g
                            g_pose = grasps_in_world[i]
                            pre_g_joint_values = j
                            g_index = i
                            previous_g_index = g_index
                            break
            #################################################################################################

        # did not find a reachable pre-grasp
        if pre_g_pose is None:
            rospy.loginfo("object out of range!")
            continue
        rospy.loginfo("grasp planning takes {}".format(time.time()-c))

        #### move to pre-grasp pose
        looking_ahead = 3
        rospy.loginfo("trying to reach {}-th pre-grasp pose".format(g_index))
        c = time.time()
        rospy.loginfo("previous trajectory is reaching: {}".format(mc.seq))

        if pre_position_trajectory is None:
            position_trajectory = mc.plan_arm_joint_values(goal_joint_values=pre_g_joint_values)
        else:
            # lazy replan
            if np.linalg.norm(np.array(current_pose[0]) - np.array(mc.get_arm_eef_pose()[0])) > 0.7:
                print(np.linalg.norm(np.array(current_pose[0]) - np.array(mc.get_arm_eef_pose()[0])))
                rospy.loginfo("eef position is still far from target position; do not replan; keep executing previous plan")
                continue
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
        if not args.ONLY_TRACKING:
            if args.conveyor_velocity == 0.03:
                start_grasp = can_grasp(pre_g_pose[0], 0.04, None)
            elif args.conveyor_velocity == 0.05:
                start_grasp = can_grasp(pre_g_pose[0], 0.06, None)
            elif args.conveyor_velocity == 0.01:
                start_grasp = can_grasp(pre_g_pose[0], 0.03, None)
            if start_grasp:
                if args.DYNAMIC:
                    rospy.loginfo("start grasping")
                    # predicted_pose = predict(1, target_object)
                    predicted_pose = tf_conversions.toTf(tf_conversions.fromMsg(motion_predict_svr(duration=1).prediction.pose))
                    if args.ONLINE_PLANNING:
                        g_pose = convert_grasps([g_pose], pose_old, predicted_pose)[0]
                    else:
                        g_pose = get_world_grasps([grasps[g_index]], args.target_object_id, old_ee_to_new_ee_translation_rotation, predicted_pose)[0][0]
                    j = mc.get_arm_ik(g_pose, avoid_collisions=False)
                    if j is None:
                        # do not check g_pose is reachable or not during grasp planning, but check predicted g_pose directly
                        # wanna drive the arm to a pre-grasp pose as soon as possible
                        rospy.loginfo("the predicted g_pose is actually not reachable, will continue")
                        continue
                    mc.move_arm_joint_values(j, plan=False) # TODO sometimes this motion is werid? rarely
                    if args.conveyor_velocity == 0.01:
                        time.sleep(1)
                    else:
                        time.sleep(0.8) # NOTE give sometime to move before closing - this is IMPORTANT, increase success rate!
                    mc.close_gripper()
                    mc.cartesian_control(z=0.05)
                    # NOTE: The trajectory returned by this will have a far away first waypoint to jump to
                    # and I assume it is because the initial position is not interpreted as valid by moveit
                    # or the good first waypoint is blocked by a instantly updated block scene
                    # mc.move_arm_joint_values(mc.HOME)
                    break
                else:
                    mc.grasp(pre_g_pose, args.DYNAMIC)
                    break
        print("\n")

    # end while
    rospy.sleep(1)  # give some time for lift to finish before get time
    time_spent = time.time() - start_time
    rospy.loginfo("time spent: {}".format(time_spent))

    # check success and then do something
    success = is_success(args.target_object_id, original_height)
    rospy.loginfo("success: {}".format(success))

    # save result to file: object_name, success, time, velocity, conveyor_distance
    result = {'object_name': args.object_name,
              'success': success,
              'time_spent': time_spent,
              'video_filename': video_fname,
              'conveyor_velocity': args.conveyor_velocity,
              'conveyor_distance': args.conveyor_distance}

    result_file_path = os.path.join(args.result_dir, '{}.csv'.format(args.object_name))
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

    # while 1:
    #     time.sleep(1)