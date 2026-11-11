"""Velocity interface on terrain: analytic reward, fall and privileged terrain features.

Rewards follow mjlab's Mjlab-Velocity-Rough-Terrain-Scan-Vel-Only-Unitree-G1 (legacy 2026-05-13 reward set,
legacy_scan_vel_only_rewards.py) as functions of the state, so the simulator labels and the world-model
imagination use the same head (paper App. A). Foot contact is soft (sole-pad clearance above the ground).
Terms and weights:
  track_linear_velocity 2.0 (std 0.5, base frame), track_angular_velocity 2.0 (std 0.5), upright torso 1.0
  and pelvis 1.0 (std sqrt(0.2)), pose 2.0 (variable posture, legacy stds; joints of tracked upper-body groups
  are scored by the upper-body command instead), body_ang_vel -0.05 (torso xy), angular_momentum -0.02 (gated
  by |yaw cmd| < 0.05), dof_pos_limits -1.0, action_rate -0.1, foot_clearance -2.0 (target 0.1 m),
  foot_slip -0.1 (with contact-foot yaw rate), feet_edge -1.0 (footprint height range > 4 cm, levels > 2),
  foothold -1.0 (stairs: footprint samples > 4 cm below the sole), stuck -1.0, cheat -1.0 (stairs:
  |heading| > 0.2), feet_flat_ori -0.4; plus the upper-body command tracking of exp7 (+1.5).
Not reproduced: air_time (weight 0 in the reference), foot_swing_height (-0.25, needs the swing-peak memory of a
contact sensor), soft_landing (-1e-5, contact forces), self_collisions (-1, contact forces; self-collision
physics is on).
Fall: pelvis tilt > 70 deg (mjlab bad_orientation) or pelvis < 0.3 m above the ground.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import mujoco
import numpy as np
import torch

from .mdp import UPPER, N_UPPER, cmd_dim
from .motion import CONTROL_DT
from .robot import G1_JOINT_NAMES, MJCF_PATH
from .rotations import yaw_mat

SCAN_SHAPE = (17, 11)  # mjlab terrain_scan: 1.6 x 1.0 m at 0.1 m, yaw-aligned, around the pelvis
SCAN_SIZE = (1.6, 1.0)
N_PADS = 4
TERRAIN_FEAT_DIM = SCAN_SHAPE[0] * SCAN_SHAPE[1] + 2 * N_PADS
FOOT_SITE = (0.04, 0.0, -0.037)  # mjlab g1.xml left/right_foot site in *_ankle_roll_link
PADS = ((-0.05, 0.02, -0.01), (-0.05, -0.02, -0.01), (0.12, 0.02, -0.01), (0.12, -0.02, -0.01))  # proxy sole pads
PAD_RADIUS = 0.025
STAIRS, FLAT = 1.0, 0.0  # terrain class flag stored per transition (info[:, 0]); info[:, 1] = terrain level

_STD = {  # legacy pose stds (standing 0.05 everywhere)
    "walking": {r".*hip_pitch.*": 0.3, r".*hip_roll.*": 0.15, r".*hip_yaw.*": 0.15, r".*knee.*": 0.35,
                r".*ankle_pitch.*": 0.25, r".*ankle_roll.*": 0.1, r".*waist_yaw.*": 0.2, r".*waist_roll.*": 0.08,
                r".*waist_pitch.*": 0.1, r".*shoulder_pitch.*": 0.15, r".*shoulder_roll.*": 0.15,
                r".*shoulder_yaw.*": 0.1, r".*elbow.*": 0.15, r".*wrist.*": 0.3},
    "running": {r".*hip_pitch.*": 0.5, r".*hip_roll.*": 0.2, r".*hip_yaw.*": 0.2, r".*knee.*": 0.6,
                r".*ankle_pitch.*": 0.35, r".*ankle_roll.*": 0.15, r".*waist_yaw.*": 0.3, r".*waist_roll.*": 0.08,
                r".*waist_pitch.*": 0.2, r".*shoulder_pitch.*": 0.5, r".*shoulder_roll.*": 0.2,
                r".*shoulder_yaw.*": 0.15, r".*elbow.*": 0.35, r".*wrist.*": 0.3},
}  # fmt: skip


@dataclass
class TerrainRewardCfg:
    w_lin: float = 2.0
    w_yaw: float = 2.0
    track_std: float = 0.5
    w_upright: float = 1.0
    upright_std: float = math.sqrt(0.2)
    w_pose: float = 2.0
    walking_threshold: float = 0.05
    running_threshold: float = 1.5
    w_body_ang_vel: float = -0.05
    w_angmom: float = -0.02
    w_joint_limit: float = -1.0
    w_action_rate: float = -0.1
    w_clearance: float = -2.0
    clearance_target: float = 0.1
    w_slip: float = -0.1
    w_edge: float = -1.0
    edge_threshold: float = 0.04
    edge_level: int = 2
    w_foothold: float = -1.0
    foothold_eps: float = 0.04
    w_stuck: float = -1.0
    w_cheat: float = -1.0
    cheat_yaw: float = 0.2
    w_feet_flat: float = -0.4
    # stair swing clearance with lookahead (mjlab tracking scandot_stair_cfg ``stairs_swing_clearance_cost``): the
    # lowest sole corner of a moving swing foot must clear the highest ground under the foot footprint and under
    # the footprint shifted `swing_lookahead` m along the foot's horizontal velocity by `swing_margin` m;
    # cost per foot = pseudo-Huber(shortfall / margin) * min(speed / swing_speed_scale, 1), swing feet on stair tiles
    # with a moving command only
    w_swing_clear: float = 0.0
    swing_margin: float = 0.05
    swing_lookahead: float = 0.08
    foot_half_length: float = 0.105
    foot_half_width: float = 0.037
    swing_lateral_margin: float = 0.03
    swing_speed_scale: float = 0.3
    # only the half of the footprint box ahead of the foot (along its horizontal velocity) plus the lookahead: the
    # trailing half over the tread being left would otherwise forbid lowering the foot on a descent until the whole
    # sole is past the edge
    swing_front_only: bool = False
    # shin clearance (stair tiles, moving command): points along each knee->ankle segment (excluding the ankle end)
    # must stay `shin_margin` m above the ground below them after a `shin_radius` capsule; the heightfield makes a
    # point that has entered a step (riser on the way up, the edge behind on the way down) a negative clearance;
    # cost = pseudo-Huber(shortfall / margin) summed over points and legs
    w_shin_clear: float = 0.0
    shin_margin: float = 0.03
    shin_radius: float = 0.05
    edge_stairs_only: bool = False  # feet_edge on stair tiles only (a sloped footprint spans > 4 cm above ~12 deg)
    w_upper: float = 1.5
    upper_std: float = 0.3
    contact_clearance: float = 0.015  # soft contact: sigmoid((contact_clearance - pad clearance) / contact_width)
    contact_width: float = 0.005
    fall_tilt_deg: float = 70.0
    fall_height: float = 0.3


def _per_joint(table: dict) -> list[float]:
    return [next(v for k, v in table.items() if re.fullmatch(k, n)) for n in G1_JOINT_NAMES]


class TerrainTask:
    """Needs kin (G1Kinematics), the scene ground (HeightSampler.ground) and the MDP (for the upper command)."""

    def __init__(self, kin, ground, mdp, device: str, cfg: TerrainRewardCfg = TerrainRewardCfg()):
        self.kin, self.ground, self.mdp, self.cfg, self.dev = kin, ground, mdp, cfg, device
        names = kin.body_names
        self.pelvis, self.torso = names.index("pelvis"), names.index("torso_link")
        self.foot_ids = [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]
        self.knee_ids = [names.index("left_knee_link"), names.index("right_knee_link")]
        self.ankle_ids = [names.index("left_ankle_pitch_link"), names.index("right_ankle_pitch_link")]
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)  # noqa: E731
        self.foot_site = t(FOOT_SITE)
        self.pads = t(PADS)
        gx = torch.linspace(-SCAN_SIZE[0] / 2, SCAN_SIZE[0] / 2, SCAN_SHAPE[0], device=device)
        gy = torch.linspace(-SCAN_SIZE[1] / 2, SCAN_SIZE[1] / 2, SCAN_SHAPE[1], device=device)
        self.scan_grid = torch.cartesian_prod(gx, gy)  # (187, 2)
        fx, fy = torch.linspace(-0.09, 0.09, 7, device=device), torch.tensor([-0.02, 0.02], device=device)
        self.footprint = torch.cartesian_prod(fx, fy)  # (14, 2): mjlab foot_height_scan 0.18 x 0.04 at 0.03
        self.swing_grid = torch.cartesian_prod(torch.linspace(0, 1, 9, device=device), torch.linspace(0, 1, 3, device=device))  # (27, 2)
        hl, hw = cfg.foot_half_length, cfg.foot_half_width
        self.sole_corners = t([[-hl, -hw], [-hl, hw], [hl, -hw], [hl, hw]])  # (4, 2) in the foot frame, at site height
        self.std = {"standing": torch.full((29,), 0.05, device=device),
                    "walking": t(_per_joint(_STD["walking"])), "running": t(_per_joint(_STD["running"]))}  # fmt: skip
        self._mass_model()
        self.max_step_reward = (cfg.w_lin + cfg.w_yaw + 2 * cfg.w_upright + cfg.w_pose
                                + (cfg.w_upper if mdp.upper_cmd else 0.0)) * CONTROL_DT  # fmt: skip
        self.cos_fall = math.cos(math.radians(cfg.fall_tilt_deg))
        if torch.device(device).type == "cuda":
            # FK of all bodies + ground lookups are ~5x faster fused; static shapes (each call site uses a fixed
            # batch size; dynamic-shape kernels were several times slower), one compiled graph per call site
            torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
            self.features = torch.compile(self.features, dynamic=False)
            self.reward = torch.compile(self.reward, dynamic=False)
            self.critic_features = torch.compile(self.critic_features, dynamic=False)
            self.soft_fall = torch.compile(self.soft_fall, dynamic=False)

    def _mass_model(self):
        """Per kin-body mass, CoM offset and inertia (fixed children such as head_link folded into their parent)."""
        m = mujoco.MjModel.from_xml_path(MJCF_PATH)
        names = self.kin.body_names
        mass, com, inertia = np.zeros(len(names)), np.zeros((len(names), 3)), np.zeros((len(names), 3, 3))
        for b in range(1, m.nbody):
            # walk up to the nearest moving body, accumulating the fixed offset
            k, pos, quat = b, np.zeros(3), np.array([1.0, 0, 0, 0])
            while m.body(k).name not in names:
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, m.body_quat[k])
                pos = m.body_pos[k] + R.reshape(3, 3) @ pos
                q2 = np.zeros(4)
                mujoco.mju_mulQuat(q2, m.body_quat[k], quat)
                quat, k = q2, m.body_parentid[k]
            i = names.index(m.body(k).name)
            Rb = np.zeros(9)
            mujoco.mju_quat2Mat(Rb, quat)
            Ri = np.zeros(9)
            q3 = np.zeros(4)
            mujoco.mju_mulQuat(q3, quat, m.body_iquat[b])
            mujoco.mju_quat2Mat(Ri, q3)
            Ri = Ri.reshape(3, 3)
            c = pos + Rb.reshape(3, 3) @ m.body_ipos[b]
            mb = m.body_mass[b]
            new_mass = mass[i] + mb
            new_com = (mass[i] * com[i] + mb * c) / max(new_mass, 1e-9)
            inertia[i] += Ri @ np.diag(m.body_inertia[b]) @ Ri.T  # parallel-axis shift neglected for fixed children
            mass[i], com[i] = new_mass, new_com
        t = lambda x: torch.tensor(x, dtype=torch.float32, device=self.dev)  # noqa: E731
        self.mass, self.com, self.inertia = t(mass), t(com), t(inertia)

    # ------------------------------------------------------------------ geometry
    def bodies(self, S):
        p, R, v, w, q, qd = S
        return self.kin.forward(p, R, v, w, q, qd, all_bodies=True)

    def feet(self, S, B=None):
        """Foot site position/velocity, foot rotation and angular velocity, pad positions (B, 2, ...)."""
        bp, bR, bv, bw = self.bodies(S) if B is None else B
        P, Rf, V, W = bp[:, self.foot_ids], bR[:, self.foot_ids], bv[:, self.foot_ids], bw[:, self.foot_ids]
        r_site = (Rf @ self.foot_site.unsqueeze(-1)).squeeze(-1)
        site = P + r_site
        site_v = V + torch.cross(W, r_site, dim=-1)
        pads = P.unsqueeze(-2) + (Rf.unsqueeze(-3) @ self.pads.unsqueeze(-1)).squeeze(-1)  # (B, 2, 4, 3)
        return site, site_v, Rf, W, pads

    def pad_clearance(self, pads):
        return pads[..., 2] - PAD_RADIUS - self.ground(pads[..., :2])  # (B, 2, 4)

    def contact(self, clearance):
        """Soft foot contact (B, 2) from the lowest sole-pad clearance."""
        c = self.cfg
        return torch.sigmoid((c.contact_clearance - clearance.min(-1).values) / c.contact_width)

    def footprint_ground(self, site, S):
        """Ground heights (B, 2, 14) on a yaw-aligned 0.18 x 0.04 m grid under each foot site."""
        Hxy = yaw_mat(S[1])[:, :2, :2]
        off = self.footprint @ Hxy.transpose(-1, -2)  # (B, 14, 2)
        return self.ground(site[:, :, None, :2] + off[:, None])

    def features(self, S):
        """(ground under the pelvis (B,), privileged terrain features (B, TERRAIN_FEAT_DIM)) for the world model."""
        p, R = S[0], S[1]
        Hxy = yaw_mat(R)[:, :2, :2]
        xy = p[:, None, :2] + self.scan_grid @ Hxy.transpose(-1, -2)
        scan = p[:, 2:3] - self.ground(xy)  # pelvis height above each scan point (mjlab height_scan convention)
        gz = self.ground(p[:, :2])
        _, _, _, _, pads = self.feet(S)
        clr = self.pad_clearance(pads).flatten(1)
        return gz, torch.cat((scan, clr), -1)

    def angular_momentum(self, B):
        bp, bR, bv, bw = B
        Rc = (bR @ self.com.unsqueeze(-1)).squeeze(-1)
        c = bp + Rc
        vc = bv + torch.cross(bw, Rc, dim=-1)
        M = self.mass.sum()
        c0 = (self.mass[:, None] * c).sum(-2, keepdim=True) / M
        v0 = (self.mass[:, None] * vc).sum(-2, keepdim=True) / M
        L = (self.mass[:, None] * torch.cross(c - c0, vc - v0, dim=-1)).sum(-2)
        Iw = bR @ self.inertia @ bR.transpose(-1, -2)
        return L + (Iw @ bw.unsqueeze(-1)).squeeze(-1).sum(-2)

    def swing_clearance(self, site, site_v, Rf):
        """(clearance (B, 2), horizontal foot speed (B, 2)): lowest sole corner minus the highest ground under the
        footprint box (plus a lateral margin) stretched `swing_lookahead` m along the foot's horizontal velocity."""
        c = self.cfg
        cy, sy = Rf[..., 0, 0], Rf[..., 1, 0]
        nrm = (cy**2 + sy**2).clamp(min=1e-6).sqrt()
        cy, sy = cy / nrm, sy / nrm  # foot yaw
        vx, vy = site_v[..., 0], site_v[..., 1]
        lx, ly = cy * vx + sy * vy, -sy * vx + cy * vy  # velocity in the foot-yaw frame
        speed = (lx**2 + ly**2 + 1e-8).sqrt()
        ax, ay = c.swing_lookahead * lx / speed.clamp(min=1e-3), c.swing_lookahead * ly / speed.clamp(min=1e-3)
        hl, hw = c.foot_half_length, c.foot_half_width + c.swing_lateral_margin
        x0, x1 = -hl + ax.clamp(max=0), hl + ax.clamp(min=0)
        if c.swing_front_only:
            ux = lx / speed.clamp(min=1e-3)
            x0, x1 = x0 + hl * ux.clamp(min=0), x1 - hl * (-ux).clamp(min=0)
        y0, y1 = -hw + ay.clamp(max=0), hw + ay.clamp(min=0)
        gx = x0[..., None] + (x1 - x0)[..., None] * self.swing_grid[:, 0]  # (B, 2, 27)
        gy = y0[..., None] + (y1 - y0)[..., None] * self.swing_grid[:, 1]
        xy = site[..., None, :2] + torch.stack((cy[..., None] * gx - sy[..., None] * gy, sy[..., None] * gx + cy[..., None] * gy), -1)
        # max(...).values, not amax: the inductor-compiled amax backward over these ground samples returns NaN
        # (tied maxima on flat treads); max's index-based backward is exact
        h = self.ground(xy).max(-1).values
        # sole corners at site height in the foot frame -> world z offsets R[2, :2] @ (x, y)
        dz = (Rf[..., 2:3, :2] @ self.sole_corners.T).squeeze(-2)  # (B, 2, 4)
        lowest = site[..., 2] + dz.min(-1).values
        return lowest - h, speed

    # ------------------------------------------------------------------ reward / fall
    def reward(self, S, cmd, a, a_prev, info, terms: bool = False):
        """info: (B, 2) = [stairs flag, terrain level]."""
        c, mdp = self.cfg, self.mdp
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        B = self.bodies(S)
        bp, bR, bv, bw = B
        vb, wb = (Rt @ v.unsqueeze(-1)).squeeze(-1), (Rt @ w.unsqueeze(-1)).squeeze(-1)
        e_lin = ((cmd[:, :2] - vb[:, :2]) ** 2).sum(-1)
        e_yaw = (cmd[:, 2] - wb[:, 2]) ** 2
        g_torso = -bR[:, self.torso, 2, :]  # world gravity in the torso frame
        g_pelvis = -R[:, 2, :]
        speed = cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()
        std = torch.where((speed < c.walking_threshold)[:, None], self.std["standing"],
                          torch.where((speed < c.running_threshold)[:, None], self.std["walking"], self.std["running"]))  # fmt: skip
        pose_w = torch.ones_like(q)
        if mdp.upper_cmd:  # tracked waist/arm groups are scored by the upper-body command term
            pose_w = torch.cat((pose_w[:, : UPPER.start], 1 - mdp.upper_mask(cmd)), -1)
        e_pose = ((q - mdp.default_q) ** 2 / std**2 * pose_w).sum(-1) / pose_w.sum(-1).clamp(min=1)
        active = (speed > 0.05).float()
        site, site_v, Rf, Wf, pads = self.feet(S, B)
        clr = self.pad_clearance(pads)
        con = self.contact(clr)
        fp = self.footprint_ground(site, S)  # (B, 2, 14)
        foot_h = site[..., 2] - fp.max(-1).values  # sole-site height above the highest ground under the foot
        clearance = ((foot_h - c.clearance_target).abs() * site_v[..., :2].norm(dim=-1)).sum(-1) * active
        slip = (con * ((site_v[..., :2] ** 2).sum(-1) + Wf[..., 2] ** 2)).sum(-1) * active
        stairs, level = info[:, 0] > 0.5, info[:, 1]
        edge = (con * ((fp.max(-1).values - fp.min(-1).values) > c.edge_threshold).float()).sum(-1) * (level > c.edge_level).float()
        if c.edge_stairs_only:
            edge = edge * stairs.float()
        foothold = (con * ((site[..., 2:3] - fp) > c.foothold_eps).float().mean(-1)).sum(-1) * stairs.float()
        stuck = ((vb[:, 0].abs() < 0.1) & (cmd[:, 0].abs() > 0.1)).float()
        heading = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        cheat = ((heading.abs() > c.cheat_yaw) & stairs).float()
        g_feet = -Rf[..., 2, :]  # gravity in each foot frame
        flat = (con * g_feet[..., :2].norm(dim=-1)).sum(-1)
        L = self.angular_momentum(B)
        swing_clear = torch.zeros_like(stuck)
        if c.w_swing_clear:
            sclr, sspeed = self.swing_clearance(site, site_v, Rf)
            # pseudo-Huber of the relative shortfall instead of mjlab's clamp to 1: quadratic for small deficits,
            # linear for deep ones, so a toe well below the next tread still gets a lifting gradient
            short = ((c.swing_margin - sclr) / c.swing_margin).clamp(min=0)
            moving_cmd = (cmd[:, :2].norm(dim=-1) > 0.1).float() * stairs.float()
            cost = 2 * ((1 + short**2).sqrt() - 1)
            swing_clear = (cost * (sspeed / c.swing_speed_scale).clamp(max=1) * (1 - con)).sum(-1) * moving_cmd
        shin_clear = torch.zeros_like(stuck)
        if c.w_shin_clear:
            pk, pa = bp[:, self.knee_ids], bp[:, self.ankle_ids]  # (B, 2, 3)
            tt = torch.tensor([0.0, 0.25, 0.5, 0.75], device=pk.device)
            pts = pk[:, :, None] + (pa - pk)[:, :, None] * tt[:, None]  # (B, 2, 4, 3)
            scl = pts[..., 2] - self.ground(pts[..., :2]) - c.shin_radius
            sshort = ((c.shin_margin - scl) / c.shin_margin).clamp(min=0)
            moving = (cmd[:, :2].norm(dim=-1) > 0.1).float() * stairs.float()
            shin_clear = (2 * ((1 + sshort**2).sqrt() - 1)).sum((-1, -2)) * moving
        r = (c.w_lin * torch.exp(-e_lin / c.track_std**2) + c.w_yaw * torch.exp(-e_yaw / c.track_std**2)
             + c.w_upright * torch.exp(-(g_torso[:, :2] ** 2).sum(-1) / c.upright_std**2)
             + c.w_upright * torch.exp(-(g_pelvis[:, :2] ** 2).sum(-1) / c.upright_std**2)
             + c.w_pose * torch.exp(-e_pose)
             + c.w_body_ang_vel * (bw[:, self.torso, :2] ** 2).sum(-1)
             + c.w_angmom * (L**2).sum(-1) * (cmd[:, 2].abs() < 0.05).float()
             + c.w_joint_limit * mdp.joint_limit(q) + c.w_action_rate * ((a - a_prev) ** 2).sum(-1)
             + c.w_clearance * clearance + c.w_slip * slip + c.w_edge * edge + c.w_foothold * foothold
             + c.w_stuck * stuck + c.w_cheat * cheat + c.w_feet_flat * flat + c.w_swing_clear * swing_clear + c.w_shin_clear * shin_clear)  # fmt: skip
        if mdp.upper_cmd:
            r = r + c.w_upper * torch.exp(-mdp.upper_err(q, cmd)[0] / c.upper_std**2)
        r = r * CONTROL_DT
        if terms:
            return r, {"e_lin": e_lin, "e_yaw": e_yaw, "contact": con, "foot_h": foot_h, "edge": edge,
                       "foothold": foothold, "stuck": stuck, "cheat": cheat, "swing_clear": swing_clear, "shin_clear": shin_clear}  # fmt: skip
        return r

    def height(self, S):
        return S[0][:, 2] - self.ground(S[0][:, :2])

    def fall(self, S) -> torch.Tensor:
        return (S[1][:, 2, 2] < self.cos_fall) | (self.height(S) < self.cfg.fall_height)

    def soft_fall(self, S, tz: float = 0.02, tt: float = 0.04) -> torch.Tensor:
        alive = torch.sigmoid((self.height(S) - self.cfg.fall_height) / tz) * torch.sigmoid((S[1][:, 2, 2] - self.cos_fall) / tt)
        return 1 - alive

    def critic_features(self, S, cmd, xi_n, terr):
        """Velocity critic input on terrain: the flat-ground features (invariant state above the ground) plus
        the privileged terrain features and soft contacts, appended so a flat critic is a prefix."""
        from .mdp import invariant_features

        gz, feat = terr
        e_vel = (self.mdp.heading_vel(S) - cmd[..., :3]) / self.mdp.cmd_scale
        parts = [invariant_features(S, gz), self.mdp.norm_cmd(cmd), e_vel]
        if self.mdp.upper_cmd:
            parts.append((S[4][..., UPPER] - cmd[..., 3 : 3 + N_UPPER]) * self.mdp.upper_mask(cmd))
        return torch.cat(parts + [xi_n, feat], -1)


def terrain_critic_dim(upper_cmd: bool) -> int:
    from .mdp import vel_critic_dim

    return vel_critic_dim(upper_cmd) + TERRAIN_FEAT_DIM


assert cmd_dim(True) == 23
