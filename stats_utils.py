import numpy as np
import tf_conversions


def get_grasp_switch_idxs(selected_grasp_indexes):

    # planned_grasps_file = data['selected_grasp_indexes']
    selected_grasp_indexes = [val for val in selected_grasp_indexes if val is not None]

    g_idxs = np.array(selected_grasp_indexes)
    idx_changes = np.where(g_idxs[:-1] != g_idxs[1:])[0]

    num_grasp_switches = len(idx_changes)

    if len(idx_changes):
        idx_changes = np.append(idx_changes, idx_changes[-1]+1)
    grasp_switch_indices = g_idxs[idx_changes]

    return num_grasp_switches, grasp_switch_indices


def get_grasp_distance(grasp_pose_list, grasp_switch_indices):
    position_distances = []
    orientation_distances = []
    for i in range(len(grasp_switch_indices)-1):
        gp_1 = grasp_pose_list[grasp_switch_indices[i]]
        gp_2 = grasp_pose_list[grasp_switch_indices[i+1]]

        pose_diff = tf_conversions.fromMsg(gp_1).Inverse()*tf_conversions.fromMsg(gp_2)     # order matters
        position_diff, orientation_diff = tf_conversions.toTf(pose_diff)

        position_distances.append(np.linalg.norm(position_diff))
        orientation_distances.append(tf_conversions.Rotation.Quaternion(*orientation_diff).GetRotAngle()[0])

    return position_distances, orientation_distances


if __name__ == "__main__":
    from geometry_msgs.msg import Pose, Point, Quaternion

    grasp_pose_list = [Pose(Point(1, 0, 0), Quaternion(0, 0, 0, 1)),
                       Pose(Point(0, 0, 1), Quaternion(0, 0, 0, 1)),
                       Pose(Point(0, 0, 1), Quaternion(1, 0, 0, 0))]
    grasp_switch_indices = [0, 1, 2]
    position_distances, orientation_distances = get_grasp_distance(grasp_pose_list, grasp_switch_indices)

    print(position_distances, orientation_distances)
