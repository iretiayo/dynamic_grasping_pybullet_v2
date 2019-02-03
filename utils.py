from collections import namedtuple
import pybullet as p

def remove_all_constraints():
    for cid in range(p.getNumConstraints()):
        p.removeConstraint(cid)

def reset_body_base(body, pose_2d):
    p.resetBasePositionAndOrientation(body, pose_2d[0], pose_2d[1])

BodyInfo = namedtuple('BodyInfo', ['base_name', 'body_name'])

def get_body_infos():
    """ Return all body info in a list """
    return [get_body_info(i) for i in range(p.getNumBodies())]


def get_body_names():
    """ Return all body names in a list """
    return [bi.body_name for bi in get_body_infos()]


def get_body_id(name):
    return get_body_names().index(name)


def get_body_info(body):
    return BodyInfo(*p.getBodyInfo(body))


def get_base_name(body):
    return get_body_info(body).base_name.decode(encoding='UTF-8')

def get_body_name(body):
    return get_body_info(body).body_name.decode(encoding='UTF-8')

def reset_camera(yaw=50.0, pitch=-35.0, dist=5.0, target=(0.0, 0.0, 0.0)):
    p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch, cameraTargetPosition=target)

def get_body_pose(body):
    """ return pose_2d """
    raw = p.getBasePositionAndOrientation(body)
    position = list(raw[0])
    orn = list(raw[1])
    return [position, orn]
