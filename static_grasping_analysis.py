import os
import time
import csv
import numpy as np
import trimesh
import argparse
import pickle
from collections import OrderedDict

import rospkg
import rospy
import tf_conversions

import pybullet as p
import pybullet_data

from mico_pybullet_simple import MicoControllerSimple
from mico_moveit import MicoMoveit
from grasp_utils import generate_grasps, get_transform, convert_graspit_pose_in_object_to_moveit_grasp_pose
from grasp_evaluation_eef_only import Controller


UNIFORM_GRASP_PLANNER = 0
SIMANN_GRASP_PLANNER = 1
REACHABILITY_AWARE_GRASP_PLANNER = 2


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('-o', '--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('-x', '--object_location_x', type=float, default=0.001,
                        help="X-location of object with respect to the robot. Ex: 0.001")
    parser.add_argument('-y', '--object_location_y', type=float, default=-0.4,
                        help="Y-location of object with respect to the robot. Ex: -0.4")
    parser.add_argument('-g', '--grasp_planning_type', type=int, default=UNIFORM_GRASP_PLANNER,
                        help="grasp planning type. Ex: 0")
    parser.add_argument('-e', '--experiment_params_fname', type=str, default='experiment_params.yaml',
                        help="Config file for experiment params. Ex: experiment_params.yaml")
    parser.add_argument('-rd', '--result_dir', type=str, default='result_dir',
                        help="Directory to store results. Ex: result_dir")
    args = parser.parse_args()

    experiment_id = '{}_{}'.format(args.object_name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    args.video_dir = os.path.join(args.result_dir, 'videos')
    if not os.path.exists(args.video_dir):
        os.makedirs(args.video_dir)
    args.video_filepath = os.path.join(args.video_dir, '{}.mp4'.format(experiment_id))

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    args.result_filepath = os.path.join(args.result_dir, 'result.csv')
    args.grasp_poses_filepath = os.path.join(args.result_dir, '{}_grasps.pk'.format(experiment_id))

    args.mesh_dir = os.path.abspath('dynamic_grasping_assets/models')
    args.gripper_urdf = os.path.abspath('dynamic_grasping_assets/mico_hand/mico_hand.urdf')
    return args


def create_scene():
    p.connect(p.GUI_SERVER)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetSimulation()
    p.resetDebugVisualizerCamera(cameraDistance=0.9, cameraYaw=-24.4, cameraPitch=-47.0,
                                 cameraTargetPosition=(-0.2, -0.30, 0.0))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # load plane, conveyor and target
    p.loadURDF("plane.urdf")

    # load robot
    urdf_dir = os.path.join(rospkg.RosPack().get_path('kinova_description'), 'urdf')
    urdf_filename = 'mico.urdf'
    if not os.path.exists(os.path.join(urdf_dir, urdf_filename)):
        os.system('cp model/mico.urdf {}'.format(os.path.join(urdf_dir, urdf_filename)))
    mico_id = p.loadURDF(os.path.join(urdf_dir, urdf_filename), flags=p.URDF_USE_SELF_COLLISION)

    pybullet_object_ids = {'robot_id': mico_id}
    return pybullet_object_ids


def create_object_urdf(object_mesh_filepath, object_name,
                       urdf_template_filepath='model/object_template.urdf',
                       urdf_target_object_filepath='model/target_object.urdf'):
    # set_up urdf
    os.system('cp {} {}'.format(urdf_template_filepath, urdf_target_object_filepath))
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name.obj', object_mesh_filepath, urdf_target_object_filepath)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name', object_name, urdf_target_object_filepath)
    os.system(sed_cmd)
    return urdf_target_object_filepath


def load_object(object_mesh_filepath, object_name, x, y):
    urdf_target_object_filepath = create_object_urdf(object_mesh_filepath, object_name)

    # position, orientation = tf_conversions.toTf(tf_conversions.fromMsg(pose))

    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    target_position = [x, y, -target_mesh.bounds.min(0)[2] + 0.01]
    orientation = [0, 0, 0, 1]
    conveyor_id = p.loadURDF("model/conveyor.urdf", [target_position[0], target_position[1], 0.01])
    target_object_id = p.loadURDF(urdf_target_object_filepath, target_position, orientation)

    pybullet_object_ids = {'conveyor_id': conveyor_id,
                           'target_object_id': target_object_id}
    return pybullet_object_ids


def create_scene_moveit(mico_moveit, pybullet_object_ids_ALL, target_object_name, target_object_mesh_filepath):
    mico_moveit.clear_scene()

    target_object_pos_ori = p.getBasePositionAndOrientation(pybullet_object_ids_ALL['target_object_id'])
    conveyor_pos_ori = p.getBasePositionAndOrientation(pybullet_object_ids_ALL['conveyor_id'])

    mico_moveit.add_mesh(target_object_name, target_object_pos_ori, target_object_mesh_filepath)
    mico_moveit.add_box("conveyor", conveyor_pos_ori, size=(.1, .1, .02))
    mico_moveit.add_box("floor", ((0, 0, -0.055), (0, 0, 0, 1)), size=(2, 2, 0.1))


def get_grasps(object_mesh_filepath_ply, planning_type=SIMANN_GRASP_PLANNER, object_pose=None, floor_offset=None):
    if planning_type == UNIFORM_GRASP_PLANNER:
        grasp_results = generate_grasps(object_mesh=object_mesh_filepath_ply, object_pose=object_pose,
                                        uniform_grasp=True)

    if planning_type == SIMANN_GRASP_PLANNER:
        grasp_results = generate_grasps(object_mesh=object_mesh_filepath_ply, object_pose=object_pose,
                                        floor_offset=floor_offset, max_steps=70000,
                                        search_energy='GUIDED_POTENTIAL_QUALITY_ENERGY', uniform_grasp=False)

    if planning_type == REACHABILITY_AWARE_GRASP_PLANNER:
        grasp_results = generate_grasps(object_mesh=object_mesh_filepath_ply, object_pose=object_pose,
                                        floor_offset=floor_offset, max_steps=70000,
                                        search_energy='REACHABLE_FIRST_HYBRID_GRASP_ENERGY', uniform_grasp=True)
    return grasp_results


def adjust_plan_to_starting_joint_range(plan, start_joint_values):
    diff = np.array(start_joint_values) - np.array(plan.joint_trajectory.points[0].positions)
    for p in plan.joint_trajectory.points:
        p.positions = (np.array(p.positions) + diff).tolist()
    return plan


def get_motion_plan_for_grasp(eef_pose, mc, mico_moveit):
    start_joint_values = mc.get_arm_joint_values()
    gripper_joint_values = mc.get_gripper_joint_values()

    pose_2d = tf_conversions.toTf(tf_conversions.fromMsg(eef_pose))

    goal_joint_values = mico_moveit.get_arm_ik(pose_2d, timeout=0.01, avoid_collisions=True,
                                               arm_joint_values=start_joint_values,
                                               gripper_joint_values=gripper_joint_values)

    if goal_joint_values is None:
        return None
    start_joint_values_converted = mico_moveit.convert_range(start_joint_values)
    goal_joint_values_converted = mico_moveit.convert_range(goal_joint_values)

    plan = mico_moveit.plan(start_joint_values_converted, goal_joint_values_converted, maximum_planning_time=0.5)

    # check if there exists a plan
    if len(plan.joint_trajectory.points) > 0:
        adjust_plan_to_starting_joint_range(plan, start_joint_values)
    return plan


def get_motion_plan_for_grasp_cartesian(eef_pose, mc, mico_moveit):
    start_joint_values = mc.get_arm_joint_values()
    start_joint_values_converted = mico_moveit.convert_range(start_joint_values)

    plan, fraction = mico_moveit.plan_straight_line(start_joint_values_converted, eef_pose, avoid_collisions=False)
    # TODO: avoid_collisions should be allow touch object

    # check if there exists a plan
    if len(plan.joint_trajectory.points) > 0:
        adjust_plan_to_starting_joint_range(plan, start_joint_values)
    return plan


def execute_grasp(mc, mico_moveit, pre_grasp_pose, grasp_pose):
    # move to pregrasp
    motion_plan = get_motion_plan_for_grasp(pre_grasp_pose, mc, mico_moveit)
    if motion_plan is None:
        return False
    mc.execute_arm_motion_plan(motion_plan)
    rospy.sleep(2)

    # move to grasp
    motion_plan = get_motion_plan_for_grasp_cartesian(grasp_pose, mc, mico_moveit)
    if motion_plan is None:
        return False
    mc.execute_arm_motion_plan(motion_plan)

    # close gripper
    mc.move_gripper_joint_values(mc.CLOSE_GRIPPER)

    # lift
    grasp_pose.position.z += 0.07
    motion_plan = get_motion_plan_for_grasp_cartesian(grasp_pose, mc, mico_moveit)
    grasp_pose.position.z -= 0.07
    if motion_plan is None:
        return False
    mc.execute_arm_motion_plan(motion_plan)
    return True


if __name__ == "__main__":
    args = get_args()

    # create grasping scene in pybullet
    pybullet_object_ids_ALL = {}
    pybullet_object_ids = create_scene()
    pybullet_object_ids_ALL.update(pybullet_object_ids)

    rospy.init_node("demo")
    p.setRealTimeSimulation(1)
    rospy.sleep(2)  # time for objects to stabilize

    # initialize robot to Home
    mc = MicoControllerSimple(pybullet_object_ids['robot_id'])
    mc.reset_arm_joint_values(mc.HOME)
    mc.reset_gripper_joint_values(mc.OPEN_GRIPPER)

    # load object
    object_name = args.object_name
    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(object_name), '{}.obj'.format(object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    pybullet_object_ids = load_object(object_mesh_filepath, object_name,
                                      x=args.object_location_x, y=args.object_location_y)
    pybullet_object_ids_ALL.update(pybullet_object_ids)

    mico_moveit = MicoMoveit()
    create_scene_moveit(mico_moveit, pybullet_object_ids_ALL, object_name, object_mesh_filepath)

    # generate grasps
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]
    target_object_pos_ori = p.getBasePositionAndOrientation(pybullet_object_ids_ALL['target_object_id'])
    object_pose = tf_conversions.toMsg(tf_conversions.fromTf(target_object_pos_ori))
    grasp_results = get_grasps(object_mesh_filepath_ply, planning_type=args.grasp_planning_type,
                               object_pose=object_pose, floor_offset=floor_offset)
    graspit_grasps, graspit_grasp_poses_in_world, graspit_grasp_poses_in_object = grasp_results

    old_ee_to_new_ee_translation_rotation = get_transform("m1n6s200_link_6", "m1n6s200_end_effector")

    # evaluate grasps
    logging = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, args.video_filepath)
    grasp_lift_results = []
    lift_motion_found = []
    stateId = p.saveState()
    for grasp_idx in range(len(graspit_grasps)):
        mc.set_arm_joint_values(joint_values=mc.HOME)
        mc.set_gripper_joint_values(mc.OPEN_GRIPPER)
        p.restoreState(stateId)

        grasp_pose, pre_grasp_pose = convert_graspit_pose_in_object_to_moveit_grasp_pose(
            graspit_grasp_poses_in_object[grasp_idx], object_pose,
            old_ee_to_new_ee_translation_rotation)

        lift_executed = execute_grasp(mc, mico_moveit, pre_grasp_pose, grasp_pose)
        rospy.sleep(2)

        # check lift success
        final_object_pos_ori = p.getBasePositionAndOrientation(pybullet_object_ids_ALL['target_object_id'])
        success = (final_object_pos_ori[0][2] - target_object_pos_ori[0][2]) > 0.02

        grasp_lift_results.append(success)
        lift_motion_found.append(lift_executed)

    # evaluate grasp using hand only
    p.setRealTimeSimulation(0)
    p.restoreState(stateId)
    p.removeBody(pybullet_object_ids_ALL['robot_id'])
    gripper_initial_pose = [[0, 0, 0.5], [0, 0, 0, 1]]
    hand = p.loadURDF(args.gripper_urdf, gripper_initial_pose[0], gripper_initial_pose[1], flags=p.URDF_USE_SELF_COLLISION)
    hand_controller = Controller(hand)
    eef_only_lift_results = []
    stateId = p.saveState()
    for grasp_idx in range(len(graspit_grasps)):
        p.restoreState(stateId)
        hand_controller.open_gripper()
        hand_controller.execute_grasp(graspit_grasp_poses_in_world[grasp_idx])
        success = (final_object_pos_ori[0][2] - target_object_pos_ori[0][2]) > 0.01
        eef_only_lift_results.append(success)
    p.stopStateLogging(logging)

    # collate results
    result = [('object_name', args.object_name),
              ('object_location_x', args.object_location_x),
              ('object_location_y', args.object_location_y),
              ('eef_only_lift_results', np.mean(eef_only_lift_results)),
              ('mean_lift_success', np.mean(grasp_lift_results)),
              ('mean_lift_motion_found', np.mean(lift_motion_found)),
              ('mean_lift_motion_found_and_success', np.mean(grasp_lift_results)/np.mean(lift_motion_found)),
              ('num_grasps', len(grasp_lift_results)),
              ('num_lift_success', np.sum(grasp_lift_results)),
              ('grasp_poses_filename', args.grasp_poses_filepath),
              ('video_filename', args.video_filepath)
              ]
    result = OrderedDict(result)
    file_exists = os.path.exists(args.result_filepath)
    with open(args.result_filepath, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    pickle.dump({'grasps': graspit_grasp_poses_in_object, 'lift_success': grasp_lift_results,
                 'eef_only_lift_results': eef_only_lift_results},
                open(args.grasp_poses_filepath, "wb"))
