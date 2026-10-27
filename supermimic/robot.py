"""Unitree G1 (29 DoF) model for MuJoCo-Warp, following BeyondMimic's actuator model.

The URDF is the one the LAFAN1 motions were retargeted to (g1_29dof_rev_1_0). Joint drives are
MuJoCo position actuators (PD, integrated implicitly) with a 10 Hz natural frequency and damping
ratio 2 on the reflected motor armature; the action scale is 0.25 * effort_limit / stiffness.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from itertools import product
from typing import TYPE_CHECKING, Literal
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .terrain import Terrain

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
URDF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof_rev_1_0.urdf")
MJCF_PATH = os.path.join(DATA_DIR, "g1_description", "g1_29dof.xml")

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _kp(arm):
    return arm * NATURAL_FREQ**2


def _kd(arm):
    return 2.0 * DAMPING_RATIO * arm * NATURAL_FREQ


# Joint order of the LAFAN1 retarget CSVs (Unitree SDK order).
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# BeyondMimic's tracked bodies; the anchor/root is the pelvis.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

# Per-joint (effort limit, velocity limit, armature) by motor type.
_M5020 = (25.0, 37.0, ARMATURE_5020)
_M4010 = (5.0, 22.0, ARMATURE_4010)
JOINT_MOTORS = {
    "hip_pitch": (88.0, 32.0, ARMATURE_7520_14),
    "hip_roll": (139.0, 20.0, ARMATURE_7520_22),
    "hip_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "knee": (139.0, 20.0, ARMATURE_7520_22),
    "ankle_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "ankle_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_yaw": (88.0, 32.0, ARMATURE_7520_14),
    "waist_roll": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "waist_pitch": (50.0, 37.0, 2.0 * ARMATURE_5020),
    "shoulder_pitch": _M5020,
    "shoulder_roll": _M5020,
    "shoulder_yaw": _M5020,
    "elbow": _M5020,
    "wrist_roll": _M5020,
    "wrist_pitch": _M4010,
    "wrist_yaw": _M4010,
}


def motor_of(joint_name: str) -> tuple[float, float, float]:
    key = joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")
    return JOINT_MOTORS[key]


def action_scale(joint_name: str) -> float:
    effort, _, arm = motor_of(joint_name)
    return 0.25 * effort / _kp(arm)


DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


def default_joint_pos_list() -> list[float]:
    out = []
    for name in G1_JOINT_NAMES:
        val = 0.0
        for pat, v in DEFAULT_JOINT_POS.items():
            if re.fullmatch(pat, name):
                val = v
        out.append(val)
    return out


def sim_model(timestep: float = 0.005, terrain: Terrain | None = None,
              collision: Literal["mesh", "proxy"] = "mesh") -> mujoco.MjModel:
    """Free pelvis, position actuators, armature, plane or terrain; no self-collisions."""
    if collision not in ("mesh", "proxy"):
        raise ValueError(f"Unknown collision mode: {collision}")
    xml = open(MJCF_PATH).read()
    acts = []
    for name in G1_JOINT_NAMES:
        effort, _, arm = motor_of(name)
        xml = re.sub(rf'(<joint name="{name}")', rf'\1 armature="{arm}"', xml, count=1)
        acts.append(f'<position name="{name}" joint="{name}" kp="{_kp(arm)}" kv="{_kd(arm)}" '
                    f'forcelimited="true" forcerange="{-effort} {effort}"/>')  # fmt: skip
    floor = '<geom name="floor" type="plane" size="0 0 0.05" contype="0" conaffinity="1" condim="3"/>'
    xml = xml.replace("<worldbody>", "<worldbody>\n    " + floor, 1)
    xml = xml.replace("</mujoco>", "  <actuator>\n    " + "\n    ".join(acts) + "\n  </actuator>\n</mujoco>", 1)
    if terrain is not None:
        from .terrain import add_hfield, write_hfield

        xml = add_hfield(xml, terrain)
    m = _compile(xml)
    if collision == "proxy":
        m = _compile(_proxy_xml(xml, m))
    if terrain is not None:
        write_hfield(m, terrain)
    robot = m.geom_bodyid > 0
    m.geom_conaffinity[robot] = 0  # robot geoms collide with the floor only
    m.opt.timestep = timestep
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    m.opt.iterations, m.opt.ls_iterations = 10, 20
    return m


def _proxy_xml(xml: str, model: mujoco.MjModel) -> str:
    """Fit primitives in link coordinates; explicit URDF inertials stay untouched.

    Mesh vertices are in MuJoCo's recentered principal frame, so transform them
    through the compiled geom pose before fitting. Visual meshes are retained.
    """
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    # MuJoCo numbers world geoms first even when the XML floor follows bodies.
    geoms = list(world.findall("geom")) + [g for b in world.findall("body") for g in b.iter("geom")]
    parents = {child: parent for parent in root.iter() for child in parent}
    for g, geom in enumerate(geoms):
        if model.geom_bodyid[g] == 0 or model.geom_contype[g] == 0:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            # 25 mm sole pads, preserving the bottom plane and a 9 cm width.
            # A 10 ms contact time constant prevents a pad center entering the
            # hfield prism, where generic convex CCD can return a bad witness.
            pos = model.geom_pos[g].copy()
            pos[2] += .025 - model.geom_size[g, 0]
            pos[1] = np.clip(pos[1], -.02, .02)
            geom.set("size", "0.025")
            geom.set("pos", " ".join(map(str, pos)))
            geom.set("solref", "0.01 1")
            continue
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[g]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[g])
        vertices = model.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + model.geom_pos[g]
        lo, hi = vertices.min(0), vertices.max(0)
        half = (hi - lo) / 2
        body = model.body(model.geom_bodyid[g]).name
        for attr in ("mesh", "quat", "pos", "euler", "axisangle"):
            geom.attrib.pop(attr, None)
        if any(name in body for name in ("pelvis", "torso", "head")):
            geom.set("type", "box")
            # The 50-contact cap applies to primitives too. Bound each box's
            # surface footprint on the 5 cm hfield even when lying on its side.
            counts = np.maximum(1, np.ceil(2 * half / .12).astype(int))
            for number, index in enumerate(product(*(range(n) for n in counts))):
                part = geom if number == 0 else deepcopy(geom)
                part.set("pos", " ".join(map(str, lo + (2 * np.asarray(index) + 1) * half / counts)))
                part.set("size", " ".join(map(str, half / counts)))
                if number:
                    parents[geom].append(part)
        else:
            # Principal mesh axes avoid inflating the radius of slanted limbs.
            vertices = model.mesh_vert[start:start + count]
            lo, hi = vertices.min(0), vertices.max(0)
            center, half = (hi + lo) / 2, (hi - lo) / 2
            axis = int(half.argmax())
            radius = float(np.max(np.delete(half, axis)))
            span = max(float(half[axis]) - radius, 0.001)
            geom.set("type", "capsule")
            geom.set("size", str(radius))
            pieces = max(1, int(np.ceil(2 * span / .10)))
            for number in range(pieces):
                part = geom if number == 0 else deepcopy(geom)
                a, b = center.copy(), center.copy()
                a[axis] += -span + 2 * span * number / pieces
                b[axis] += -span + 2 * span * (number + 1) / pieces
                ends = np.stack((a, b)) @ rotation.reshape(3, 3).T + model.geom_pos[g]
                part.set("fromto", " ".join(map(str, ends.ravel())))
                if number:
                    parents[geom].append(part)
    return ET.tostring(root, encoding="unicode")


def _compile(xml: str) -> mujoco.MjModel:
    # compile next to the MJCF so relative mesh paths resolve
    tmp = os.path.join(os.path.dirname(MJCF_PATH), f"_sim_{os.getpid()}.xml")
    open(tmp, "w").write(xml)
    try:
        return mujoco.MjModel.from_xml_path(tmp)
    finally:
        os.remove(tmp)
