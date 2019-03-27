from __future__ import division
import pybullet as p
import pybullet_data
from mico_pybullet import MicoController
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
from geometry_msgs.msg import Pose, Point, Quaternion
from graspit_interface.msg import Grasp

from grasp_utils import load_reachability_params, get_transform, get_reachability_of_grasps_pose, generate_grasps, \
    create_occupancy_grid_from_obstacles, convert_graspit_pose_in_object_to_moveit_grasp_pose


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('-o', '--object_name', type=str, default= 'cube',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('-v', '--conveyor_velocity', type=float, default=0.05,
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
    args.target_mesh_bounds = target_mesh.bounds
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

    args.ONLY_TRACKING = True
    args.DYNAMIC = True
    args.KEEP_PREVIOUS_GRASP = True
    args.RANK_BY_REACHABILITY = True
    args.LOAD_OBSTACLES = True
    args.ONLINE_PLANNING = False

    args.scene_fnm = "scene.yaml"
    args.scene_config = yaml.load(open(args.scene_fnm))

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


def can_grasp(grasp_pose, object_pose, eef_pose, d_gpos_threshold=None, d_target_threshold=None):
    grasp_position = [grasp_pose.position.x, grasp_pose.position.y, grasp_pose.position.z]
    target_position = [object_pose.position.x, object_pose.position.y, object_pose.position.z]
    eef_position = [eef_pose.position.x, eef_pose.position.y, eef_pose.position.z]

    # TODO shall I give constraint on quaternion as well?
    d_target = np.linalg.norm(np.array(target_position) - np.array(eef_position))
    d_gpos = np.linalg.norm(np.array(grasp_position) - np.array(eef_position))
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
    mc.mico_moveit.add_box("floor", ((0, 0, -0.055), (0, 0, 0, 1)), size=(2, 2, 0.1))

    if args.LOAD_OBSTACLES:
        for obstacle in args.scene_config['obstacles']:
            mc.mico_moveit.add_mesh(obstacle['object_name'], ut.get_body_pose(obstacle['body_id']), obstacle['obj_filepath'])
        time.sleep(1)


def get_reachability_space(args):
    rospy.loginfo("start creating sdf reachability space...")
    start_time = time.time()
    if args.LOAD_OBSTACLES:
        sdf_reachability_space_fnm = os.path.splitext(args.scene_fnm)[0]+'_'+str(args.conveyor_distance)+'_reach_data.sdf'
        sdf_reachability_space_filepath = os.path.join(args.reachability_data_dir, sdf_reachability_space_fnm)
        if os.path.exists(sdf_reachability_space_filepath):
            # load sdf for this scene if already exists
            sdf_reachability_space = np.fromfile(sdf_reachability_space_filepath, dtype=float)
            sdf_reachability_space = sdf_reachability_space.reshape(args.dims)
        else:
            # create reachability space from obstacles. requires masking out obstacle areas
            obstacle_mesh_filepaths, obstacle_poses = [], []
            for obstacle in args.scene_config['obstacles']:
                obstacle_mesh_filepaths.append(obstacle['ply_filepath'])
                obstacle_poses.append(ut.list_2_pose(ut.get_body_pose(obstacle['body_id'])))

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
            sdf_reachability_space.tofile(open(sdf_reachability_space_filepath, 'w'))
    else:
        _, mins, step_size, dims, sdf_reachability_space = load_reachability_data_from_dir(args.reachability_data_dir)
    rospy.loginfo("sdf reachability space created, which takes {}".format(time.time()-start_time))
    return sdf_reachability_space


if __name__ == "__main__":
    args = get_args()
    rospy.set_param("start_moving", False)
    create_scence(args)

    rospy.init_node("demo")

    # initialize controller
    mc = MicoController(args.mico_id)
    mc.reset_arm_joint_values(mc.HOME)

    # starting pose
    mc.move_arm_joint_values(mc.HOME, plan=False)
    mc.open_gripper()

    create_scence_moveit(args, mc)

    # load grasps
    starting_x = -0.6
    if args.ONLINE_PLANNING:
        # need to make sure seed grasp is good!
        grasp_filepath = "grasps_online_" + str(args.conveyor_distance) + '.pk'
        object_pose_init = Pose(
            Point(starting_x, args.conveyor_distance, ut.get_body_pose(args.target_object_id)[0][2]),
            Quaternion(0, 0, 0, 1))
        search_energy = 'REACHABLE_FIRST_HYBRID_GRASP_ENERGY'
        max_steps = 35000
    else:
        grasp_filepath = os.path.join(args.mesh_dir, args.object_name, "grasps_" + str(args.object_name) + '.pk')
        object_pose_init = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))
        search_energy = 'GUIDED_POTENTIAL_QUALITY_ENERGY'
        max_steps = 70000
    grasp_results = generate_grasps(load_fnm=grasp_filepath, save_fnm=grasp_filepath,
                                    object_mesh=args.object_mesh_filepath_ply,
                                    object_pose=object_pose_init, floor_offset=args.target_mesh_bounds.min(0)[2],
                                    max_steps=max_steps, search_energy=search_energy, seed_grasp=None)
    graspit_grasps, graspit_grasp_poses_in_world, graspit_grasp_poses_in_object = grasp_results

    # create sdf reachability space
    if args.RANK_BY_REACHABILITY:
        sdf_reachability_space = get_reachability_space(args)

    # kalman filter service
    rospy.wait_for_service('motion_prediction')
    motion_predict_svr = rospy.ServiceProxy('motion_prediction', motion_prediction.srv.GetFuturePose)
    rospy.set_param("start_moving", True)

    pre_position_trajectory = None

    video_fname = '{}-{}.mp4'.format(args.object_name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, os.path.join(args.video_dir, video_fname))

    #############################################################################################
    current_grasp_idx = None # previous grasp pose index
    original_height = ut.get_body_pose(args.target_object_id)[0][2]
    old_ee_to_new_ee_translation_rotation = get_transform("m1n6s200_link_6", "m1n6s200_end_effector")
    new_ee_to_old_ee_translation_rotation = get_transform("m1n6s200_end_effector", "m1n6s200_link_6")
    #
    # if args.ONLINE_PLANNING:
    #     graspit_grasp_pose_in_object = graspit_grasp_poses_in_object[0]
    #     pose_old = object_pose_init

    while ut.get_body_pose(args.target_object_id)[0][0] < starting_x:
        pass
    start_time = time.time()    # enter workspace

    while True:
        current_pose = ut.get_body_pose(args.target_object_id)
        current_conveyor_pose = ut.get_body_pose(args.conveyor_id)

        if current_conveyor_pose[0][0] > args.max_x-0.05:
            # target moves outside workspace, break directly
            break

        # grasp planning
        grasp_planning_start = time.time()
        future_pose = motion_predict_svr(duration=1).prediction.pose

        # information about the current grasp
        pre_g_pose, g_pose, pre_g_joint_values, current_grasp_idx = None, None, None, None

        if args.ONLINE_PLANNING:
            ee_in_world, pre_g_pose = convert_graspit_pose_in_object_to_moveit_grasp_pose(graspit_grasp_poses_in_object[0],
                                                                                          future_pose,
                                                                                          old_ee_to_new_ee_translation_rotation)
            j = mc.get_arm_ik(pre_g_pose)
            if j is not None: # previous grasp after conversion is still reachabale
                rospy.loginfo("the previous pre-grasp is still reachabale")
                pre_g_joint_values = j
            else:
                rospy.loginfo("online planning... finding new grasps")
                seed_grasp = Grasp()
                seed_grasp.pose = graspit_grasp_poses_in_world[0]
                grasp_results = generate_grasps(object_mesh=args.object_mesh_filepath_ply,
                                                object_pose=future_pose, floor_offset=args.target_mesh_bounds.min(0)[2],
                                                max_steps=30000 + 50, seed_grasp=seed_grasp,
                                                search_energy='REACHABLE_FIRST_HYBRID_GRASP_ENERGY')
                graspit_grasps, graspit_grasp_poses_in_world, graspit_grasp_poses_in_object = grasp_results
        else:
            pre_grasps_in_world, ees_in_world = [], []
            for g in graspit_grasp_poses_in_object:
                ee_in_world, pre_g_pose = convert_graspit_pose_in_object_to_moveit_grasp_pose(g, future_pose,
                                                                                              old_ee_to_new_ee_translation_rotation)
                ees_in_world.append(ee_in_world)
                pre_grasps_in_world.append(pre_g_pose)

            if args.KEEP_PREVIOUS_GRASP and current_grasp_idx is not None:
                pre_g_joint_values = mc.get_arm_ik(tf_conversions.toTf(tf_conversions.fromMsg(pre_grasps_in_world[current_grasp_idx])))
                if pre_g_joint_values is not None:   # ik_exists
                    rospy.loginfo("the previous pre-grasp is reachable")
                else:
                    current_grasp_idx = None

            if current_grasp_idx is None:
                grasp_order_idxs = range(len(pre_grasps_in_world))
                if args.RANK_BY_REACHABILITY:
                    sdf_values = get_reachability_of_grasps_pose(graspit_grasp_poses_in_world,
                                                                 sdf_reachability_space, args.mins, args.step_size,
                                                                 args.dims)
                    grasp_order_idxs = np.argsort(sdf_values)[::-1]
                for idx in range(len(grasp_order_idxs)):
                    g_idx = grasp_order_idxs[idx]
                    pre_g_joint_values = mc.get_arm_ik(tf_conversions.toTf(tf_conversions.fromMsg(pre_grasps_in_world[g_idx])))
                    if pre_g_joint_values is not None:
                        current_grasp_idx = g_idx
                        break

            # did not find a reachable pre-grasp
            if current_grasp_idx is None:
                rospy.loginfo("object out of range! Or no grasp reachable")
                continue
            else:
                ee_in_world, pre_g_pose = ees_in_world[current_grasp_idx], pre_grasps_in_world[current_grasp_idx]
        rospy.loginfo("grasp planning takes {}".format(time.time()-grasp_planning_start))

        #### move to pre-grasp pose
        looking_ahead = 3
        rospy.loginfo("trying to reach {}-th pre-grasp pose".format(current_grasp_idx))
        motion_start = time.time()
        rospy.loginfo("previous trajectory is reaching: {}".format(mc.seq))

        if pre_position_trajectory is None:
            position_trajectory, motion_plan = mc.plan_arm_joint_values(goal_joint_values=pre_g_joint_values)
            # position_trajectory, motion_plan = mc.plan_arm_eef_pose(ee_pose=pre_g_pose)
        else:
            # lazy replan
            if np.linalg.norm(np.array(current_pose[0]) - np.array(mc.get_arm_eef_pose()[0])) > 0.7:
                print(np.linalg.norm(np.array(current_pose[0]) - np.array(mc.get_arm_eef_pose()[0])))
                rospy.loginfo("eef position is still far from target position; do not replan; keep executing previous plan")
                continue
            else:
                # start_index = min(mc.seq + looking_ahead, len(pre_position_trajectory) - 1)
                # start_joint_values = pre_position_trajectory[start_index]
                # start_joint_values = mc.get_arm_joint_values()
                planning_start_time = rospy.Time.now()
                time_since_start = (planning_start_time - mc.start_time_stamp).to_sec()
                planning_time = 0.25
                start_joint_values = mc.interpolate_plan_at_time(motion_plan, time_since_start + planning_time)
                position_trajectory, motion_plan = mc.plan_arm_joint_values(goal_joint_values=pre_g_joint_values,
                                                                            start_joint_values=start_joint_values)
                # position_trajectory, motion_plan = mc.plan_arm_eef_pose(ee_pose=pre_g_pose,
                #                                                         start_joint_values=start_joint_values)
                sleep_time = planning_time - (rospy.Time.now() - planning_start_time).to_sec()
                print('Sleep after planning: {}'.format(sleep_time))
                rospy.sleep(max(0, sleep_time))
        rospy.loginfo("planning takes {}".format(time.time()-motion_start))

        if position_trajectory is None:
            rospy.loginfo("No plans found!")
        else:
            rospy.loginfo("start executing")
            pre_position_trajectory = position_trajectory # just another reference
            mc.execute_arm_trajectory(position_trajectory, motion_plan)
            time.sleep(0.2)

        # TODO sometimes grasp planning takes LONGER with some errors after tracking for a long time, This results the previous
        # trajectory to have finished before we send another goal to move arm
        # TODO add blocking to control

        #### grasp
        eef_pose = tf_conversions.toMsg(tf_conversions.fromTf(mc.get_arm_eef_pose()))
        # eef_pose = tf_conversions.toMsg(tf_conversions.fromTf(get_transform(mc.mico_moveit.BASE_LINK, mc.mico_moveit.TIP_LINK)))
        object_pose = tf_conversions.toMsg(tf_conversions.fromTf(p.getBasePositionAndOrientation(args.target_object_id)))
        if not args.ONLY_TRACKING:
            if args.conveyor_velocity == 0.01:
                d_gpos_threshold = 0.03
            elif args.conveyor_velocity == 0.03:
                d_gpos_threshold = 0.04
            elif args.conveyor_velocity == 0.05:
                d_gpos_threshold = 0.06
            start_grasp = can_grasp(pre_g_pose, object_pose, eef_pose,
                                    d_gpos_threshold=d_gpos_threshold, d_target_threshold=None)
            if start_grasp:
                if args.DYNAMIC:
                    rospy.loginfo("start grasping")
                    # predicted_pose = predict(1, target_object)
                    predicted_pose = motion_predict_svr(duration=1).prediction.pose
                    if args.ONLINE_PLANNING:
                        ee_idx = 0
                    else:
                        ee_idx = current_grasp_idx
                    ee_in_world, pre_g_pose = convert_graspit_pose_in_object_to_moveit_grasp_pose(graspit_grasp_poses_in_object[ee_idx], predicted_pose,
                                                                                                  old_ee_to_new_ee_translation_rotation)
                    j = mc.get_arm_ik(tf_conversions.toTf(tf_conversions.fromMsg(ee_in_world)), avoid_collisions=False)
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