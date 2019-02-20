from collections import namedtuple
import pybullet as p
from geometry_msgs.msg import Pose, Point, Quaternion

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