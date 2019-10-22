from collections import namedtuple
import pybullet as p
import numpy as np
import time

INF = np.inf
PI = np.pi
CIRCULAR_LIMITS = -PI, PI


def step(duration=1):
    for i in range(duration * 240):
        p.stepSimulation()


def step_real(duration=1):
    for i in range(duration * 240):
        p.stepSimulation()
        time.sleep(1.0 / 240.0)


def split_7d(pose):
    return [list(pose[:3]), list(pose[3:])]


# Constraints

ConstraintInfo = namedtuple('ConstraintInfo', ['parentBodyUniqueId', 'parentJointIndex',
                                               'childBodyUniqueId', 'childLinkIndex', 'constraintType',
                                               'jointAxis', 'jointPivotInParent', 'jointPivotInChild',
                                               'jointFrameOrientationParent', 'jointFrameOrientationChild',
                                               'maxAppliedForce'])


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


# Joints

JOINT_TYPES = {
    p.JOINT_REVOLUTE: 'revolute',  # 0
    p.JOINT_PRISMATIC: 'prismatic',  # 1
    p.JOINT_SPHERICAL: 'spherical',  # 2
    p.JOINT_PLANAR: 'planar',  # 3
    p.JOINT_FIXED: 'fixed',  # 4
    p.JOINT_POINT2POINT: 'point2point',  # 5
    p.JOINT_GEAR: 'gear',  # 6
}


def get_num_joints(body):
    return p.getNumJoints(body)


def get_joints(body):
    return list(range(get_num_joints(body)))


def get_joint(body, joint_or_name):
    if type(joint_or_name) is str:
        return joint_from_name(body, joint_or_name)
    return joint_or_name


JointInfo = namedtuple('JointInfo', ['jointIndex', 'jointName', 'jointType',
                                     'qIndex', 'uIndex', 'flags',
                                     'jointDamping', 'jointFriction', 'jointLowerLimit', 'jointUpperLimit',
                                     'jointMaxForce', 'jointMaxVelocity', 'linkName', 'jointAxis',
                                     'parentFramePos', 'parentFrameOrn', 'parentIndex'])


def get_joint_info(body, joint):
    return JointInfo(*p.getJointInfo(body, joint))


def get_joints_info(body, joints):
    return [JointInfo(*p.getJointInfo(body, joint)) for joint in joints]


def get_joint_name(body, joint):
    return get_joint_info(body, joint).jointName.decode('UTF-8')


def get_joint_names(body):
    return [get_joint_name(body, joint) for joint in get_joints(body)]


def joint_from_name(body, name):
    for joint in get_joints(body):
        if get_joint_name(body, joint) == name:
            return joint
    raise ValueError(body, name)


def has_joint(body, name):
    try:
        joint_from_name(body, name)
    except ValueError:
        return False
    return True


def joints_from_names(body, names):
    return tuple(joint_from_name(body, name) for name in names)


JointState = namedtuple('JointState', ['jointPosition', 'jointVelocity',
                                       'jointReactionForces', 'appliedJointMotorTorque'])


def get_joint_state(body, joint):
    return JointState(*p.getJointState(body, joint))


def get_joint_position(body, joint):
    return get_joint_state(body, joint).jointPosition


def get_joint_torque(body, joint):
    return get_joint_state(body, joint).appliedJointMotorTorque


def get_joint_positions(body, joints=None):
    return tuple(get_joint_position(body, joint) for joint in joints)


def set_joint_position(body, joint, value):
    p.resetJointState(body, joint, value)


def set_joint_positions(body, joints, values):
    assert len(joints) == len(values)
    for joint, value in zip(joints, values):
        set_joint_position(body, joint, value)


def get_configuration(body):
    return get_joint_positions(body, get_movable_joints(body))


def set_configuration(body, values):
    set_joint_positions(body, get_movable_joints(body), values)


def get_full_configuration(body):
    # Cannot alter fixed joints
    return get_joint_positions(body, get_joints(body))


def get_joint_type(body, joint):
    return get_joint_info(body, joint).jointType


def is_movable(body, joint):
    return get_joint_type(body, joint) != p.JOINT_FIXED


def get_movable_joints(body):  # 45 / 87 on pr2
    return [joint for joint in get_joints(body) if is_movable(body, joint)]


def joint_from_movable(body, index):
    return get_joints(body)[index]


def is_circular(body, joint):
    joint_info = get_joint_info(body, joint)
    if joint_info.jointType == p.JOINT_FIXED:
        return False
    if joint_info.jointUpperLimit < joint_info.jointLowerLimit:
        raise ValueError("circular joint, check it out!")


def get_joint_limits(body, joint):
    """
    Obtain the limits of a single joint
    :param body: int
    :param joint: int
    :return: (int, int), lower limit and upper limit
    """
    if is_circular(body, joint):
        return CIRCULAR_LIMITS
    joint_info = get_joint_info(body, joint)
    return joint_info.jointLowerLimit, joint_info.jointUpperLimit


def get_joints_limits(body, joints):
    """
    Obtain the limits of a set of joints
    :param body: int
    :param joints: array type
    :return: a tuple of 2 arrays - lower limit and higher limit
    """
    lower_limit = []
    upper_limit = []
    for joint in joints:
        lower_limit.append(get_joint_info(body, joint).jointLowerLimit)
        upper_limit.append(get_joint_info(body, joint).jointUpperLimit)
    return lower_limit, upper_limit


def get_min_limit(body, joint):
    return get_joint_limits(body, joint)[0]


def get_max_limit(body, joint):
    return get_joint_limits(body, joint)[1]


def get_max_velocity(body, joint):
    return get_joint_info(body, joint).jointMaxVelocity


def get_max_force(body, joint):
    return get_joint_info(body, joint).jointMaxForce


def get_joint_q_index(body, joint):
    return get_joint_info(body, joint).qIndex


def get_joint_v_index(body, joint):
    return get_joint_info(body, joint).uIndex


def get_joint_axis(body, joint):
    return get_joint_info(body, joint).jointAxis


def get_joint_parent_frame(body, joint):
    joint_info = get_joint_info(body, joint)
    return joint_info.parentFramePos, joint_info.parentFrameOrn


def violates_limit(body, joint, value):
    if not is_circular(body, joint):
        lower, upper = get_joint_limits(body, joint)
        if (value < lower) or (upper < value):
            return True
    return False


def violates_limits(body, joints, values):
    return any(violates_limit(body, joint, value) for joint, value in zip(joints, values))


def wrap_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def wrap_joint(body, joint, value):
    if is_circular(body, joint):
        return wrap_angle(value)
    return value


# Body and base

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


# Control

def control_joint(body, joint, value):
    return p.setJointMotorControl2(bodyUniqueId=body,
                                   jointIndex=joint,
                                   controlMode=p.POSITION_CONTROL,
                                   targetPosition=value,
                                   targetVelocity=0,
                                   maxVelocity=get_max_velocity(body, joint),
                                   force=get_max_force(body, joint))


def control_joints(body, joints, positions):
    return p.setJointMotorControlArray(body, joints, p.POSITION_CONTROL,
                                       targetPositions=positions,
                                       targetVelocities=[0.0] * len(joints),
                                       forces=[get_max_force(body, joint) for joint in joints])


# Camera

CameraInfo = namedtuple('CameraInfo', ['width', 'height',
                                       'viewMatrix', 'projectionMatrix', 'cameraUp',
                                       'cameraForward', 'horizontal', 'vertical',
                                       'yaw', 'pitch', 'dist', 'target'])


def reset_camera(yaw=50.0, pitch=-35.0, dist=5.0, target=(0.0, 0.0, 0.0)):
    p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target)


def get_camera():
    return CameraInfo(*p.getDebugVisualizerCamera())


# Visualization

def create_frame_marker(pose=((0, 0, 0), (0, 0, 0, 1)),
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
    position = np.array(pose[0])
    orientation = np.array(pose[1])

    pts = np.array([[0, 0, 0], [line_length, 0, 0], [0, line_length, 0], [0, 0, line_length]])
    rotIdentity = np.array([0, 0, 0, 1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0, :], rotIdentity)
    px, _ = p.multiplyTransforms(position, orientation, pts[1, :], rotIdentity)
    py, _ = p.multiplyTransforms(position, orientation, pts[2, :], rotIdentity)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3, :], rotIdentity)

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


def create_arrow_marker(pose=((0, 0, 0), (0, 0, 0, 1)),
                        line_length=0.1,
                        arrow_length=0.01,
                        line_width=2,
                        arrow_width=6,
                        life_time=0,
                        color_index=0,
                        replace_frame_id=None):
    """
    Create an arrow marker that identifies the z-axis of the end effector frame. Add a dot towards the positive direction.
    """

    position = np.array(pose[0])
    orientation = np.array(pose[1])

    pts = np.array([[0, 0, 0], [line_length, 0, 0], [0, line_length, 0], [0, 0, line_length]])
    z_extend = [0, 0, line_length + arrow_length]
    rotIdentity = np.array([0, 0, 0, 1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0, :], rotIdentity)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3, :], rotIdentity)
    pz_extend, _ = p.multiplyTransforms(position, orientation, z_extend, rotIdentity)

    if replace_frame_id is not None:
        z_id = p.addUserDebugLine(po, pz, rgb_colors_1[color_index], line_width, life_time,
                                  replaceItemUniqueId=replace_frame_id[2])
        z_extend_id = p.addUserDebugLine(pz, pz_extend, rgb_colors_1[color_index], arrow_width, life_time,
                                         replaceItemUniqueId=replace_frame_id[2])
    else:
        z_id = p.addUserDebugLine(po, pz, rgb_colors_1[color_index], line_width, life_time)
        z_extend_id = p.addUserDebugLine(pz, pz_extend, rgb_colors_1[color_index], arrow_width, life_time)
    frame_id = (z_id, z_extend_id)
    return frame_id


# https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
rgb_colors_255 = [(230, 25, 75),  # red
                  (60, 180, 75),  # green
                  (255, 225, 25),  # yello
                  (0, 130, 200),  # blue
                  (245, 130, 48),  # orange
                  (145, 30, 180),  # purple
                  (70, 240, 240),  # cyan
                  (240, 50, 230),  # magenta
                  (210, 245, 60),  # lime
                  (250, 190, 190),  # pink
                  (0, 128, 128),  # teal
                  (230, 190, 255),  # lavender
                  (170, 110, 40),  # brown
                  (255, 250, 200),  # beige
                  (128, 0, 0),  # maroon
                  (170, 255, 195),  # lavender
                  (128, 128, 0),  # olive
                  (255, 215, 180),  # apricot
                  (0, 0, 128),  # navy
                  (128, 128, 128),  # grey
                  (0, 0, 0),  # white
                  (255, 255, 255)]  # black

rgb_colors_1 = np.array(rgb_colors_255) / 255.
