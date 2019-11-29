import pprint
from collections import OrderedDict
import os
import csv
import pybullet_data
import pybullet as p
import pybullet_utils as pu


def write_csv_line(result_file_path, result):
    """ write a line in a csv file; create the file if the line does not already exist """
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def configure_pybullet(rendering=False):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    pu.reset_camera(yaw=50.0, pitch=-35.0, dist=1.200002670288086, target=(0.0, 0.0, 0.0))
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)