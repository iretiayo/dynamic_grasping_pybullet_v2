from collections import namedtuple
import pybullet as p
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np
import rospy
import time

def print_loop_end(loop_start):
    rospy.loginfo("loop ends; whole loop takes {}".format(time.time() - loop_start))
    print("\n")

def pose_2_list(pose):
    """

    :param pose: geometry_msgs.msg.Pose
    :return: pose_2d: [[x, y, z], [x, y, z, w]]
    """
    position = [pose.position.x, pose.position.y, pose.position.z]
    orientation = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    return [position, orientation]

def list_2_pose(pose_2d):
    """

    :param pose_2d: [[x, y, z], [x, y, z, w]]
    :return: pose: geometry_msgs.msg.Pose
    """
    return Pose(Point(*pose_2d[0]), Quaternion(*pose_2d[1]))


### Constraints
ConstraintInfo = namedtuple('ConstraintInfo', ['parentBodyUniqueId', 'parentJointIndex',
                                               'childBodyUniqueId', 'childLinkIndex', 'constraintType',
                                               'jointAxis', 'jointPivotInParent', 'jointPivotInChild',
                                               'jointFrameOrientationParent', 'jointFrameOrientationChild', 'maxAppliedForce'])


def remove_all_constraints():
    for cid in get_constraint_ids():
        p.removeConstraint(cid)

def get_constraint_ids():
    """
    getConstraintUniqueId will take a serial index in range 0..getNumConstraints,  and reports the constraint unique id.
    Note that the constraint unique ids may not be contiguous, since you may remove constraints.
    """
    return sorted([p.getConstraintUniqueId(i) for i in range(p.getNumConstraints())])

def get_constraint_info(constraint):
    # there are four additional arguments
    return ConstraintInfo(*p.getConstraintInfo(constraint)[:11])

### Body and base
BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])

def remove_all_bodies():
    for i in get_body_ids():
        p.removeBody(i)

def reset_body_base(body, pose_2d):
    p.resetBasePositionAndOrientation(body, pose_2d[0], pose_2d[1])

def get_body_infos():
    """ Return all body info in a list """
    return [get_body_info(i) for i in get_body_ids()]

def get_body_names():
    """ Return all body names in a list """
    return [bi.body_name for bi in get_body_infos()]

def get_body_id(name):
    return get_body_names().index(name)

def get_body_ids():
    return sorted([p.getBodyUniqueId(i) for i in range(p.getNumBodies())])

def get_body_info(body):
    return BodyInfo(*p.getBodyInfo(body))

def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')

def get_body_name(body):
    return get_body_info(body).body_name.decode(encoding='UTF-8')

def get_body_pose(body):
    """ return pose_2d """
    raw = p.getBasePositionAndOrientation(body)
    position = list(raw[0])
    orn = list(raw[1])
    return [position, orn]

### Camera
CameraInfo = namedtuple('CameraInfo', ['width', 'height',
                                               'viewMatrix', 'projectionMatrix', 'cameraUp',
                                               'cameraForward', 'horizontal', 'vertical',
                                               'yaw', 'pitch', 'dist', 'target'])

def reset_camera(yaw=50.0, pitch=-35.0, dist=5.0, target=(0.0, 0.0, 0.0)):
    p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target)

def get_camera():
    return CameraInfo(*p.getDebugVisualizerCamera())

### Visualization
def create_frame_marker(pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
                        x_color=np.array([1, 0, 0]),
                        y_color=np.array([0, 1, 0]),
                        z_color=np.array([0, 0, 1]),
                        line_length=0.1,
                        line_width=2,
                        life_time=0,
                        replace_frame_id=None):
    """
    Create a pose marker that identifies a position and orientation in space with 3 colored lines.
    """
    pose_2d = pose_2_list(pose)
    position = np.array(pose_2d[0])
    orientation = np.array(pose_2d[1])

    pts = np.array([[0,0,0],[line_length,0,0],[0,line_length,0],[0,0,line_length]])
    rotIdentity = np.array([0,0,0,1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0,:], rotIdentity)
    px, _ = p.multiplyTransforms(position, orientation, pts[1,:], rotIdentity)
    py, _ = p.multiplyTransforms(position, orientation, pts[2,:], rotIdentity)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3,:], rotIdentity)

    if replace_frame_id is not None:
        x_id = p.addUserDebugLine(po, px, x_color, line_width, life_time, replaceItemUniqueId=replace_frame_id[0])
        y_id = p.addUserDebugLine(po, py, y_color, line_width, life_time, replaceItemUniqueId=replace_frame_id[1])
        z_id = p.addUserDebugLine(po, pz, z_color, line_width, life_time, replaceItemUniqueId=replace_frame_id[2])
    else:
        x_id = p.addUserDebugLine(po, px, x_color, line_width, life_time)
        y_id = p.addUserDebugLine(po, py, y_color, line_width, life_time)
        z_id = p.addUserDebugLine(po, pz, z_color, line_width, life_time)
    frame_id = (x_id, y_id, z_id)
    return frame_id

def create_arrow_marker(pose=Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1)),
                        line_length=0.1,
                        arrow_length=0.01,
                        line_width=2,
                        arrow_width=6,
                        life_time=0,
                        color_index = 0,
                        replace_frame_id=None):
    """
    Create an arrow marker that identifies the z-axis of the end effector frame.
    """

    pose_2d = pose_2_list(pose)
    position = np.array(pose_2d[0])
    orientation = np.array(pose_2d[1])

    pts = np.array([[0,0,0],[line_length,0,0],[0,line_length,0],[0,0,line_length]])
    z_extend = [0,0,line_length + arrow_length]
    rotIdentity = np.array([0, 0, 0, 1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0, :], rotIdentity)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3,:], rotIdentity)
    pz_extend, _ = p.multiplyTransforms(position, orientation, z_extend, rotIdentity)

    if replace_frame_id is not None:
        z_id = p.addUserDebugLine(po, pz, rgb_colors_1[color_index], line_width, life_time, replaceItemUniqueId=replace_frame_id[2])
        z_extend_id = p.addUserDebugLine(pz, pz_extend, rgb_colors_1[color_index], arrow_width, life_time, replaceItemUniqueId=replace_frame_id[2])
    else:
        z_id = p.addUserDebugLine(po, pz, rgb_colors_1[color_index], line_width, life_time)
        z_extend_id = p.addUserDebugLine(pz, pz_extend, rgb_colors_1[color_index], arrow_width, life_time)
    frame_id = (z_id, z_extend_id)
    return frame_id


# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
rgb_colors_255 = [(230, 25, 75),    # red
                  (60, 180, 75),    # green
                  (255, 225, 25),   # yello
                  (0, 130, 200),    # blue
                  (245, 130, 48),   # orange
                  (145, 30, 180),   # purple
                  (70, 240, 240),   # cyan
                  (240, 50, 230),   # magenta
                  (210, 245, 60),   # lime
                  (250, 190, 190),  # pink
                  (0, 128, 128),    # teal
                  (230, 190, 255),  # lavender
                  (170, 110, 40),   # brown
                  (255, 250, 200),  # beige
                  (128, 0, 0),      # maroon
                  (170, 255, 195),  # lavender
                  (128, 128, 0),    # olive
                  (255, 215, 180),  # apricot
                  (0, 0, 128),      # navy
                  (128, 128, 128),  # grey
                  (0, 0, 0),        # white
                  (255, 255, 255)]  # black

rgb_colors_1 = [(0.9019607843137255, 0.09803921568627451, 0.29411764705882354),   # red
                (0.23529411764705882, 0.7058823529411765, 0.29411764705882354),   # green
                (1.0, 0.8823529411764706, 0.09803921568627451),                   # yello
                (0.0, 0.5098039215686274, 0.7843137254901961),                    # blue
                (0.9607843137254902, 0.5098039215686274, 0.18823529411764706),    # orange
                (0.5686274509803921, 0.11764705882352941, 0.7058823529411765),    # purple
                (0.27450980392156865, 0.9411764705882353, 0.9411764705882353),    # cyan
                (0.9411764705882353, 0.19607843137254902, 0.9019607843137255),    # magenta
                (0.8235294117647058, 0.9607843137254902, 0.23529411764705882),    # lime
                (0.9803921568627451, 0.7450980392156863, 0.7450980392156863),     # pink
                (0.0, 0.5019607843137255, 0.5019607843137255),                    # teal
                (0.9019607843137255, 0.7450980392156863, 1.0),                    # lavender
                (0.6666666666666666, 0.43137254901960786, 0.1568627450980392),    # brown
                (1.0, 0.9803921568627451, 0.7843137254901961),                    # beige
                (0.5019607843137255, 0.0, 0.0),                                   # maroon
                (0.6666666666666666, 1.0, 0.7647058823529411),                    # lavender
                (0.5019607843137255, 0.5019607843137255, 0.0),                    # olive
                (1.0, 0.8431372549019608, 0.7058823529411765),                    # apricot
                (0.0, 0.0, 0.5019607843137255),                                   # navy
                (0.5019607843137255, 0.5019607843137255, 0.5019607843137255),     # grey
                (0.0, 0.0, 0.0),                                                  # black
                (1.0, 1.0, 1.0)]                                                  # white
