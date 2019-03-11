import tf_conversions
import tf
import tf2_ros
import tf2_kdl
import rospy
import os
import pickle
import numpy as np

from reachability_utils.reachability_resolution_analysis import interpolate_pose_in_reachability_space_grid
import plyfile


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


## TODO uniform sampling grasps

def generate_grasps(load_fnm=None, save_fnm=None, body="cube", body_extents=(0.05, 0.05, 0.05)):
    """
    This method assumes the target object to be at world origin. Filter out bad grasps by volume quality.

    :param load_fnm: load file name
    :param save_fnm: save file name. save as graspit grasps.grasps
    :param body: the name of the graspable object to load in graspit, or the mesh file path
    :param body_extents: the extents of the bounding box
    :return: graspit grasps.grasps
    """
    ## NOTE, now use sim-ann anf then switch

    if load_fnm:
        grasps = pickle.load(open(load_fnm, "rb"))
        return grasps
    else:
        gc = graspit_commander.GraspitCommander()
        gc.clearWorld()

        ## creat scene in graspit
        floor_offset = -body_extents[2] / 2 - 0.01  # half of the block size + half of the conveyor
        floor_pose = Pose(Point(-1, -1, floor_offset), Quaternion(0, 0, 0, 1))
        body_pose = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))

        gc.importRobot('MicoGripper')
        gc.importGraspableBody(body, body_pose)
        gc.importObstacle('floor', floor_pose)
        grasps = gc.planGrasps()
        grasps = grasps.grasps
        grasps = [g for g in grasps if g.volume_quality > 0]

        if save_fnm:
            pickle.dump(grasps, open(save_fnm, "wb"))
        return grasps


def plan_reachable_grasps(load_fnm=None, save_fnm=None, object_name="cube", object_pose_2d=None, seed_grasp=None,
                          max_steps=35000):
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
    gc.importObstacle('floor',
                      Pose(Point(object_pose_2d[0][0] - 2, object_pose_2d[0][1] - 2, 0), Quaternion(0, 0, 0, 1)))
    # TODO not considering conveyor?

    # simulated annealling
    # import ipdb; ipdb.set_trace()
    # grasps = gc.planGrasps(max_steps=max_steps+30000, search_energy='REACHABLE_FIRST_HYBRID_GRASP_ENERGY',
    #                        use_seed_grasp=seed_grasp is not None, seed_grasp=seed_grasp)
    grasps = gc.planGrasps(max_steps=max_steps + 30000,
                           use_seed_grasp=seed_grasp is not None, seed_grasp=seed_grasp)
    if grasps is None:
        print("here")
    grasps = grasps.grasps

    # keep only good grasps
    # TODO is this really required?
    # good_grasps = [g for g in grasps if g.volume_quality > 0]
    good_grasps = grasps

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


def get_world_grasps(grasps, objectID, old_ee_to_new_ee_translation_rotation, object_pose=None):
    """
    Also change eef link.
    :param grasps: grasps.grasps returned by graspit
    :param objectID: object id
    :param object_pose: the pose of the target; if None, use current pose. pose_2d
    :return: a list of tf tuples
    """
    if object_pose is None:
        object_pose = p.getBasePositionAndOrientation(objectID)
    world_T_object = tf_conversions.toMatrix(tf_conversions.fromTf(object_pose))
    grasps_in_world = list()
    grasps_in_world_before_eef_trans = list()
    for g in grasps:
        object_g = tf_conversions.toMatrix(tf_conversions.fromMsg(g.pose))
        world_g = world_T_object.dot(object_g)
        world_g_pose = tf_conversions.toMsg(tf_conversions.fromMatrix(world_g))
        grasps_in_world_before_eef_trans.append(tf_conversions.toTf(tf_conversions.fromMsg(world_g_pose)))
        # change end effector link
        world_g_pose_new = change_end_effector_link(world_g_pose, old_ee_to_new_ee_translation_rotation)
        grasps_in_world.append(tf_conversions.toTf(tf_conversions.fromMsg(world_g_pose_new)))
    return grasps_in_world, grasps_in_world_before_eef_trans


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


def load_reachability_params(reachability_data_dir):
    step_size = np.loadtxt(os.path.join(reachability_data_dir, 'reach_data.step'), delimiter=',')
    mins = np.loadtxt(os.path.join(reachability_data_dir, 'reach_data.mins'), delimiter=',')
    dims = np.loadtxt(os.path.join(reachability_data_dir, 'reach_data.dims'), delimiter=',', dtype=int)

    return step_size, mins, dims


def get_reachability_of_grasps_pose(grasps_in_world, sdf_reachability_space, mins, step_size, dims):
    """ grasps_in_world is a list of geometry_msgs/Pose """
    sdf_values = []
    for g_pose in grasps_in_world:
        trans, rot = tf_conversions.toTf(tf_conversions.fromMsg(g_pose))
        rpy = tf_conversions.Rotation.Quaternion(*rot).GetRPY()
        query_pose = np.concatenate((trans, rpy))
        sdf_values.append(
            interpolate_pose_in_reachability_space_grid(sdf_reachability_space, mins, step_size, dims, query_pose))

    # is_reachable = [sdf_values[i] > 0 for i in range(len(sdf_values))]
    return sdf_values


def get_reachability_of_grasps_pose_2d(grasps_in_world, sdf_reachability_space, mins, step_size, dims):
    """ grasps_in_world is a list of pose_2d """
    sdf_values = []
    for g_pose in grasps_in_world:
        trans, rot = g_pose[0], g_pose[1]
        rpy = tf_conversions.Rotation.Quaternion(*rot).GetRPY()
        query_pose = np.concatenate((trans, rpy))
        sdf_values.append(
            interpolate_pose_in_reachability_space_grid(sdf_reachability_space,
                                                        mins, step_size, dims, query_pose))

    # is_reachable = [sdf_values[i] > 0 for i in range(len(sdf_values))]
    return sdf_values


def convert_grasps(grasps, old_pose, new_pose):
    """
    Given a set of grasps (in world frame) generated when target object is at old_pose (in world frame),
    compute the switched grasps (in world frame) when target object has moved to new_pose (in world frame)

    MATH: w_T_gnew = w_T_n * o_T_w * w_T_gold
    o: old object frame, n: new object frame, w: world frame

    :param grasps: list of pose_2d grasps
    :param old_pose: pose_2d
    :param new_pose: pose_2d
    :return: list of pose_2d grasps
    """
    grasps_new = list()
    for g in grasps:
        w_T_gold = tf_conversions.toMatrix(tf_conversions.fromTf(g))
        o_T_w = tf.transformations.inverse_matrix(tf_conversions.toMatrix(tf_conversions.fromTf(old_pose)))
        w_T_n = tf_conversions.toMatrix(tf_conversions.fromTf(new_pose))
        w_T_gnew = (w_T_n.dot(o_T_w)).dot(w_T_gold)
        grasps_new.append(tf_conversions.toTf(tf_conversions.fromMatrix(w_T_gnew)))
    return grasps_new


def read_vertex_points_from_ply_filepath(ply_filepath):
    ply = plyfile.PlyData.read(ply_filepath)

    mesh_vertices = np.ones((ply['vertex']['x'].shape[0], 3))
    mesh_vertices[:, 0] = ply['vertex']['x']
    mesh_vertices[:, 1] = ply['vertex']['y']
    mesh_vertices[:, 2] = ply['vertex']['z']
    return mesh_vertices


def transform_points(vertices, transform):
    vertices_hom = np.ones((vertices.shape[0], 4))
    vertices_hom[:, :-1] = vertices

    # Create new 4xN transformed array
    transformed_vertices_hom = np.dot(transform, vertices_hom.T).T

    transformed_vertices = transformed_vertices_hom[:, :-1]

    return transformed_vertices


def add_obstacles_to_reachability_space_full(points, mins, step_size, dims):
    voxel_grid = np.zeros(shape=dims)

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)

    grid_points_min = np.round((bbox_min - np.array(mins)) / step_size).astype(int)
    grid_points_max = np.round((bbox_max - np.array(mins)) / step_size).astype(int)

    grid_points_min = np.clip(grid_points_min, 0, dims - 1)
    grid_points_max = np.clip(grid_points_max, 0, dims - 1)

    voxel_grid[grid_points_min[0]:grid_points_max[0] + 1, grid_points_min[1]:grid_points_max[1] + 1,
    grid_points_min[2]:grid_points_max[2] + 1] = 1

    return voxel_grid


def create_occupancy_grid_from_obstacles(obstacle_mesh_filepaths, obstacle_poses, mins_xyz, step_size_xyz, dims_xyz):
    voxel_grid = np.zeros(shape=dims_xyz)

    for filepath, pose in zip(obstacle_mesh_filepaths, obstacle_poses):
        vertices = read_vertex_points_from_ply_filepath(filepath)
        # if obstacle.type == 'box':
        #     sample_points = np.meshgrid(*[np.linspace(-sz / 2., sz / 2., 4) for sz in obstacle.box_size])
        #     vertices = np.array(sample_points).reshape(len(sample_points), -1).T
        frame = tf_conversions.fromMsg(pose)
        transform = tf_conversions.toMatrix(frame)
        vertices_transformed = transform_points(vertices, transform)

        voxel_grid += add_obstacles_to_reachability_space_full(vertices_transformed, mins_xyz, step_size_xyz,
                                                               dims_xyz)
    voxel_grid[np.where(voxel_grid > 0)] = 1
    return voxel_grid
