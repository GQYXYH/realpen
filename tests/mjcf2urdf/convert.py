import os.path as osp

import pybullet_utils.bullet_client as bulllet_client
import pybullet_utils.urdfEditor as urdfEditor

import os
import six


def makedirs(name, mode=0o777, exist_ok=True):
    """An wrapper of os.makedirs that accepts exist_ok."""
    name = str(name)
    if six.PY2:
        try:
            os.makedirs(name, mode)
        except OSError:
            if not os.path.isdir(name):
                raise
    else:
        os.makedirs(name, mode, exist_ok=exist_ok)


def convert_mjcf_to_urdf(input_mjcf, output_path='./'):
    """Convert MuJoCo mjcf to URDF format and save.
    Parameters
    ----------
    input_mjcf : str
        input path of mjcf file.
    output_path : str
        output directory path of urdf.
    """
    client = bulllet_client.BulletClient()
    objs = client.loadMJCF(input_mjcf, flags=client.URDF_USE_IMPLICIT_CYLINDER)

    # create output directory
    makedirs(output_path)

    for obj in objs:
        humanoid = objs[obj]
        ue = urdfEditor.UrdfEditor()
        ue.initializeFromBulletBody(humanoid, client._client)
        robot_name = str(client.getBodyInfo(obj)[1], 'utf-8')
        part_name = str(client.getBodyInfo(obj)[0], 'utf-8')
        save_visuals = False
        outpath = osp.join(
            output_path, "{}_{}.urdf".format(robot_name, part_name))
        ue.saveUrdf(outpath, save_visuals)


if __name__ == '__main__':
    convert_mjcf_to_urdf('./qube_convert.xml')