import moveit_commander
import rospy

class MicoMoveit(object):
    def __init__(self):
        self.arm_commander_group = moveit_commander.MoveGroupCommander('arm')

    def get_arm_eff_pose(self):
        """ Return [[x, y, x], [x, y, z, w]]"""
        pose_stamped = self.arm_commander_group.get_current_pose()
        pose = pose_stamped.pose
        position = [pose.position.x, pose.position.y, pose.position.z]
        orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        return [position, orientation]

if __name__ == "__main__":
    rospy.init_node('mico_moveit')

    mm = MicoMoveit()
    print(mm.get_arm_eff_pose())