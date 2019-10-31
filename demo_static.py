import os
import numpy as np
import pybullet as p
import pybullet_data
import time
import trimesh
import argparse
import grasp_utils as gu
import pybullet_utils as pu
from collections import OrderedDict
import csv
import tqdm
import tf_conversions
from mico_controller import MicoController
import rospy
import rospkg
import threading
from geometry_msgs.msg import PoseStamped

from reachability_utils.process_reachability_data_from_csv import load_reachability_data_from_dir


def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')

    parser.add_argument('--object_name', type=str, default='bleach_cleanser',
                        help="Target object to be grasped. Ex: cube")
    parser.add_argument('--grasp_database_path', type=str, default='yeah')
    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--realtime', action='store_true', default=False)
    parser.add_argument('--num_trials', type=int, default=1)
    args = parser.parse_args()

    if args.realtime:
        args.rendering = True

    args.mesh_dir = os.path.abspath('assets/models')
    args.robot_urdf = os.path.abspath('assets/mico/mico.urdf')

    args.reachability_data_dir = os.path.join(rospkg.RosPack().get_path('mico_reachability_config'), 'data')

    return args


def get_reachability_space(reachability_data_dir):
    rospy.loginfo("start creating sdf reachability space...")
    start_time = time.time()
    _, mins, step_size, dims, sdf_reachability_space = load_reachability_data_from_dir(reachability_data_dir)
    rospy.loginfo("sdf reachability space created, which takes {}".format(time.time()-start_time))
    return sdf_reachability_space, mins, step_size, dims


def configure_pybullet(rendering=False):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)


def write_csv_line(result_file_path, index, num_trials, num_successes, volume_quality, epsilon_quality, grasp_fnm):
    result = [('index', index),
              ('num_trials', num_trials),
              ('num_successes', num_successes),
              ('volume_quality', volume_quality),
              ('epsilon_quality', epsilon_quality),
              ('grasp_fnm', grasp_fnm)]
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def create_object_urdf(object_mesh_filepath, object_name,
                       urdf_template_filepath='assets/object_template.urdf',
                       urdf_target_object_filepath='assets/target_object.urdf'):
    # set_up urdf
    os.system('cp {} {}'.format(urdf_template_filepath, urdf_target_object_filepath))
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name.obj', object_mesh_filepath, urdf_target_object_filepath)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{}|g' {}".format('object_name', object_name, urdf_target_object_filepath)
    os.system(sed_cmd)
    return urdf_target_object_filepath


class DynamicGraspingWorld:

    def __init__(self,
                 target_name,
                 target_initial_pose,
                 robot_initial_pose,
                 robot_initial_state,
                 conveyor_initial_pose,
                 robot_urdf,
                 target_urdf,
                 target_mesh_file_path,
                 grasp_database_path,
                 reachability_data_dir,
                 realtime):
        self.target_name = target_name
        self.target_initial_pose = target_initial_pose
        self.robot_initial_pose = robot_initial_pose
        self.robot_initial_state = robot_initial_state
        self.conveyor_initial_pose = conveyor_initial_pose
        self.robot_urdf = robot_urdf
        self.target_urdf = target_urdf
        self.target_mesh_file_path = target_mesh_file_path
        self.realtime = realtime

        self.grasp_database_path = grasp_database_path
        self.grasp_database = np.load(os.path.join(self.grasp_database_path, self.target_name + '.npy'))
        self.reachability_data_dir = reachability_data_dir
        self.sdf_reachability_space, self.mins, self.step_size, self.dims = get_reachability_space(self.reachability_data_dir)

        self.plane = p.loadURDF("plane.urdf")
        self.target = p.loadURDF(self.target_urdf, self.target_initial_pose[0], self.target_initial_pose[1])
        self.robot = MicoController(self.robot_initial_pose, self.robot_initial_state, self.robot_urdf)
        self.conveyor = p.loadURDF("assets/conveyor.urdf", conveyor_initial_pose[0], conveyor_initial_pose[1])

        self.reset()

        self.target_pose_pub = rospy.Publisher('target_pose', PoseStamped, queue_size=1)
        self.conveyor_pose_pub = rospy.Publisher('conveyor_pose', PoseStamped, queue_size=1)
        rospy.set_param('target_mesh_file_path', self.target_mesh_file_path)

        update_scene_thread = threading.Thread(target=self.update_scene_threading)
        update_scene_thread.daemon = True
        update_scene_thread.start()

    def update_scene_threading(self):
        r = rospy.Rate(30)
        while True:
            target_pose = pu.get_body_pose(self.target)
            conveyor_pose = pu.get_body_pose(self.conveyor)
            self.target_pose_pub.publish(gu.list_2_ps(target_pose))
            self.conveyor_pose_pub.publish(gu.list_2_ps(conveyor_pose))
            rospy.set_param('pose_published', True)
            r.sleep()
            # target_pose = pu.get_body_pose(self.target)
            # conveyor_pose = pu.get_body_pose(self.conveyor)
            # self.robot.scene.add_mesh(self.target_name, gu.list_2_ps(target_pose), self.target_mesh_file_path)
            # self.robot.scene.add_box('conveyor', gu.list_2_ps(conveyor_pose), size=(.1, .1, .02))

    def reset(self):
        p.resetBasePositionAndOrientation(self.target, self.target_initial_pose[0], self.target_initial_pose[1])
        self.robot.reset()

        pu.step(2)

    def step(self, freeze_time, motion_plan):
        # calculate conveyor pose, change constraint
        # calculate arm pose, control array
        for i in range(int(freeze_time * 240)):
            self.robot.step()
            # the conveyor step here
            p.stepSimulation()
            if self.realtime:
                time.sleep(1.0/240.0)
        if motion_plan is not None:
            self.robot.update_motion_plan(motion_plan)

    def static_grasp(self):
        target_pose = pu.get_body_pose(self.target)
        predicted_pose = target_pose

        grasp_planning_time, grasp, grasp_jv = self.plan_grasp(predicted_pose, None, None)
        motion_planning_time, plan = self.plan_motion(grasp_jv)
        self.robot.execute_plan(plan, self.realtime)
        self.robot.close_gripper(self.realtime)
        plan, fraction = self.robot.plan_cartesian_control(z=0.07)
        self.robot.execute_plan(plan, self.realtime)

    def dynamic_grasp(self):
        grasp, grasp_jv = None, None
        while not False:
            target_pose = pu.get_body_pose(self.target)
            predicted_pose = target_pose

            grasp_planning_time, grasp, grasp_jv = self.plan_grasp(predicted_pose, grasp, grasp_jv)
            self.step(grasp_planning_time, None)

            motion_planning_time, plan = self.plan_motion(grasp_jv)
            self.step(motion_planning_time, plan)

    def plan_grasp(self, target_pose, old_grasp, old_grasp_jv):
        start_time = time.time()
        if old_grasp is not None:
            if self.robot.get_arm_ik(old_grasp) is not None:
                planning_time = time.time() - start_time
                print("Planning a grasp takes {:.6f}".format(planning_time))
                return planning_time, old_grasp, old_grasp_jv
        grasps_in_world = [gu.convert_grasp_in_object_to_world(target_pose, pu.split_7d(g)) for g in self.grasp_database]
        sdf_values = gu.get_reachability_of_grasps_pose_2d(grasps_in_world,
                                                           self.sdf_reachability_space,
                                                           self.mins,
                                                           self.step_size,
                                                           self.dims)
        grasp_order_idxs = np.argsort(sdf_values)[::-1]

        temp_start = time.time()
        # TODO this part can be done before grasps are added to database
        # grasps_in_world_ee = [gu.pose_2_list(gu.change_end_effector_link(gu.list_2_pose(g), gu.link6_reference_to_ee)) for g in grasps_in_world]
        grasps_in_world_ee = [gu.change_end_effector_link_pose_2d(g) for g in grasps_in_world]
        print(time.time()-temp_start)
        planned_grasp = grasps_in_world_ee[grasp_order_idxs[0]]
        planned_joint_values = self.robot.get_arm_ik(planned_grasp)
        # gu.visualize_grasps_with_reachability(grasps_in_world_ee, sdf_values)
        # gu.visualize_grasp_with_reachability(planned_grasp, sdf_values[grasp_order_idxs[0]], maximum=max(sdf_values), minimum=min(sdf_values))
        planning_time = time.time() - start_time
        print("Planning a grasp takes {:.6f}".format(planning_time))
        return planning_time, planned_grasp, planned_joint_values

    def plan_motion(self, grasp_jv):
        predicted_period = 0.2
        start_time = time.time()

        if self.robot.discretized_plan is not None:
            future_target_index = min(int(predicted_period * 240 + self.robot.wp_target_index), len(self.robot.discretized_plan)-1)
            start_joint_values = self.robot.discretized_plan[future_target_index]
            plan = self.robot.plan_arm_joint_values(grasp_jv, start_joint_values=start_joint_values)
        else:
            plan = self.robot.plan_arm_joint_values(grasp_jv)
        planning_time = time.time() - start_time

        print("Planning a motion takes {:.6f}".format(planning_time))
        return planning_time, plan


if __name__ == "__main__":
    args = get_args()
    configure_pybullet(args.rendering)
    rospy.init_node('dynamic_grasping')

    object_mesh_filepath = os.path.join(args.mesh_dir, '{}'.format(args.object_name), '{}.obj'.format(args.object_name))
    object_mesh_filepath_ply = object_mesh_filepath.replace('.obj', '.ply')
    target_urdf = create_object_urdf(object_mesh_filepath, args.object_name)
    target_mesh = trimesh.load_mesh(object_mesh_filepath)
    floor_offset = target_mesh.bounds.min(0)[2]
    target_initial_pose = [[0.3, 0.3, -target_mesh.bounds.min(0)[2] + 0.02], [0, 0, 0, 1]]
    robot_initial_pose = [[0, 0, 0], [0, 0, 0, 1]]
    conveyor_initial_pose = [[0.3, 0.3, 0.01], [0, 0, 0, 1]]

    dynamic_grasping_world = DynamicGraspingWorld(target_name=args.object_name,
                                                  target_initial_pose=target_initial_pose,
                                                  robot_initial_pose=robot_initial_pose,
                                                  robot_initial_state=MicoController.HOME,
                                                  conveyor_initial_pose=conveyor_initial_pose,
                                                  robot_urdf=args.robot_urdf,
                                                  target_urdf=target_urdf,
                                                  target_mesh_file_path=object_mesh_filepath,
                                                  grasp_database_path=args.grasp_database_path,
                                                  reachability_data_dir=args.reachability_data_dir,
                                                  realtime=args.realtime)

    for i in range(args.num_trials):
        dynamic_grasping_world.static_grasp()
        dynamic_grasping_world.reset()

    print("finished")

