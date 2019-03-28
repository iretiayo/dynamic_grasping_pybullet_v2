from collections import namedtuple
import pybullet as p
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np

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