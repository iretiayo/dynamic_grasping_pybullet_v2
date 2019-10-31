import moveit_commander as mc
import rospy
from geometry_msgs.msg import PoseStamped


class SceneUpdater:
    def __init__(self):
        self.target_pose_stamped = None
        self.conveyor_pose_stamped = None
        rospy.Subscriber("target_pose", PoseStamped, self.target_listen_cb, queue_size=1)
        rospy.Subscriber("conveyor_pose", PoseStamped, self.conveyor_listen_cb, queue_size=1)

    def target_listen_cb(self, pose_stamped):
        self.target_pose_stamped = pose_stamped

    def conveyor_listen_cb(self, pose_stamped):
        self.conveyor_pose_stamped = pose_stamped


if __name__ == "__main__":
    rospy.init_node('update_scene', anonymous=True)
    target_mesh_file_path = rospy.get_param('target_mesh_file_path', None)
    scene_updater = SceneUpdater()
    scene = mc.PlanningSceneInterface()

    while target_mesh_file_path is None or \
            scene_updater.target_pose_stamped is None or \
            scene_updater.conveyor_pose_stamped is None:
        target_mesh_file_path = rospy.get_param('target_mesh_file_path', None)
        rospy.sleep(0.1)

    while not rospy.is_shutdown():
        scene.add_mesh('shit', scene_updater.target_pose_stamped, target_mesh_file_path)
        scene.add_box('conveyor', scene_updater.conveyor_pose_stamped, size=(.1, .1, .02))
