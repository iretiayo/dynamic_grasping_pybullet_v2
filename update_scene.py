import moveit_commander as mc
import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


class SceneUpdater:
    def __init__(self):
        self.target_pose_stamped = None
        self.conveyor_pose_stamped = None
        self.target_mesh_file_path = None
        rospy.Subscriber("target_pose", PoseStamped, self.target_listen_cb, queue_size=1)
        rospy.Subscriber("conveyor_pose", PoseStamped, self.conveyor_listen_cb, queue_size=1)
        rospy.Subscriber("target_mesh", String, self.target_mesh_listen_cb, queue_size=1)

    def target_listen_cb(self, pose_stamped):
        self.target_pose_stamped = pose_stamped

    def conveyor_listen_cb(self, pose_stamped):
        self.conveyor_pose_stamped = pose_stamped

    def target_mesh_listen_cb(self, target_mesh_file_path):
        self.target_mesh_file_path = target_mesh_file_path.data


if __name__ == "__main__":
    rospy.init_node('update_scene', anonymous=True)
    scene_updater = SceneUpdater()
    scene = mc.PlanningSceneInterface()

    print('Waiting for the first msg to be published...')
    while scene_updater.target_mesh_file_path is None or \
            scene_updater.target_pose_stamped is None or \
            scene_updater.conveyor_pose_stamped is None:
        rospy.sleep(0.1)
    print('The first msg received!')

    while not rospy.is_shutdown():
        scene.add_mesh('target', scene_updater.target_pose_stamped, scene_updater.target_mesh_file_path)
        scene.add_box('conveyor', scene_updater.conveyor_pose_stamped, size=(.1, .1, .02))
