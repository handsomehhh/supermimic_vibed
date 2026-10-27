"""MuJoCo scene for replaying recorded rollouts: the simulated G1 (solid) and a reference ghost.

Geom groups: 1 robot visuals, 2 ghost (translucent blue), 3 robot collision geoms (hidden by default).
qpos layout: robot free joint + 29 joints (0:36), then the ghost's (36:72).
"""

import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
MJCF = os.path.join(ROOT, "data", "g1_description", "g1_29dof.xml")


def two_robot_xml(offwidth: int = 1920, offheight: int = 1080) -> str:
    xml = open(MJCF).read()
    xml = re.sub(r'meshdir="[^"]*"', 'meshdir="' + os.path.join(ROOT, "data", "g1_description") + '/"', xml)
    body = xml[xml.index('<body name="pelvis"') : xml.index("</worldbody>")]
    # robot collision geoms (no explicit group) go to group 3 so only the visual meshes show
    robot = re.sub(r"<geom (?![^>]*group=)", '<geom group="3" ', body)
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    ghost = re.sub(r"<geom [^>]*group=\"1\"[^>]*/>", "", ghost)  # one copy of each mesh is enough for the ghost
    ghost = re.sub(r'rgba="[^"]*"', 'rgba="0.35 0.65 1.0 0.28"', ghost)
    ghost = re.sub(r' (contype|conaffinity|group)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", '<geom contype="0" conaffinity="0" group="2" ')
    scene = """
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    """
    xml = xml.replace(body, robot).replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.92 0.93 0.94" rgb2="0.80 0.82 0.85" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.08"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.98" rgb2="0.75 0.8 0.88" width="256" height="256"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    if "<visual>" not in xml:
        xml = xml.replace("<asset>", f'<visual><global offwidth="{offwidth}" offheight="{offheight}"/><quality shadowsize="4096"/></visual>\n  <asset>', 1)
    return xml
