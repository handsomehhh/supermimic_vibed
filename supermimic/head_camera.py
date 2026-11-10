"""G1 head depth camera (D435 on torso_link) rendered in torch, for the simulator and for imagined states.

Geometry: the head D435 frame ``d435_link`` is fixed to ``torso_link`` at xyz (0.0576235, 0.01753, 0.42987),
rpy (0, 0.8307767, 0) (g1_29dof_rev_1_0.urdf; also a body in data/g1_description/g1_29dof.xml), i.e. the
optical axis (+x of the link) points forward and 47.6 deg down; it moves with the waist joints.

Image model (mjlab tracking velocity slot, ``_DEPTH_CAMERA_CFG`` without the pelvis mount): 64 x 36
pinhole, 89.04 x 57.9 deg FOV, distance to the image plane; then the same noise pipeline
  gaussian (sigma = 0.005 + 0.01 d) -> invalid below 0.25 m, clip to [0, 2] m, normalize to [0, 1],
  invalid -> -1 -> 1% dropout to -1 -> crop (top 10, bottom 0, left 10, right 10) -> bilinear resize to 36 x 64,
with per-episode extrinsic (+-2 cm, roll/yaw +-1 deg, pitch +-0.04 rad) and intrinsic (FOV +-1 deg, principal
point +-1 px) perturbations. Rays hit the scene ground (``HeightSampler.ground``: the terrain map and the
z = 0 floor around it) and the robot's own legs, feet and arms (spheres along their proxy capsules).
The camera refreshes at 30 Hz against 50 Hz control: a frame is taken at control step k (since reset,
plus a per-episode phase) whenever floor(3 (k + phase) / 5) changes; the policy sees the last 10 frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from .rotations import axis_angle_to_mat, quat_to_mat

# camera-frame perturbation parameters stored per episode: dx, dy, dz, roll, pitch, yaw, dfov (deg), dcx, dcy (px), phase,
# latency (control steps)
CAM_PARAM_DIM = 11


@dataclass
class HeadCameraCfg:
    width: int = 64
    height: int = 36
    hfov_deg: float = 89.04
    vfov_deg: float = 57.9
    mount_link: str = "torso_link"
    mount_pos: tuple[float, float, float] = (0.0576235, 0.01753, 0.42987)
    mount_pitch: float = 0.8307767239493009
    max_depth: float = 2.0
    invalid_below: float = 0.25
    noise_std: float = 0.005
    noise_std_per_m: float = 0.01
    dropout: float = 0.01
    crop: tuple[int, int, int, int] = (10, 0, 10, 10)  # top, bottom, left, right
    pos_range: tuple[float, float, float] = (0.02, 0.02, 0.02)
    roll_range: float = 0.01745
    pitch_range: float = 0.04
    yaw_range: float = 0.01745
    fov_range_deg: float = 1.0
    center_range_px: float = 1.0
    frames: int = 10
    # the policy sees `frames` arrived frames `stride` frames apart (stride 2: 10 frames spanning 0.6 s at 30 Hz);
    # the simulator keeps the last ring_len = stride * (frames - 1) + 1 arrived frames
    stride: int = 1
    # sensor realism (0 = off): per-episode latency in control steps (a frame taken at step k arrives at k + L),
    # per-frame drop probability (the frame never arrives), and invalid pixels at depth discontinuities larger
    # than edge_jump m (stereo shadows / flying pixels) with probability edge_dropout
    latency_steps: tuple[int, int] = (0, 0)
    frame_drop: float = 0.0
    edge_dropout: float = 0.0
    edge_jump: float = 0.1
    # ray marching: coarse steps of `step` m in image-plane depth from `near`, then bisection
    near: float = 0.05
    step: float = 0.1
    refine: int = 5
    self_occlusion: bool = True


def is_tick(k: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """True at control steps (since reset) where a new 30 Hz frame is taken; step 0 always has one."""
    k = k.long()
    ph = phase.long()
    return (k == 0) | ((3 * (k + ph)) // 5 != (3 * (k + ph - 1)) // 5)


def arrives(k: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """True at control steps k (since reset) where a frame arrives: one taken at k - latency (params[:, 10])."""
    lat = params[:, 10].long()
    return (k.long() >= lat) & is_tick(k.long() - lat, params[:, 9])


def ring_len(cfg: "HeadCameraCfg") -> int:
    return cfg.stride * (cfg.frames - 1) + 1


def max_latency(cfg: "HeadCameraCfg") -> int:
    return int(cfg.latency_steps[1])


def _occluder_spheres(kin_body_names: list[str]) -> tuple[list[int], np.ndarray, np.ndarray]:
    """(kin body index, local center, radius) of spheres covering legs, feet and arms (proxy capsules)."""
    from .robot import sim_model

    m = sim_model(collision="proxy")
    bodies, centers, radii = [], [], []
    keep = ("hip_roll", "hip_yaw", "knee", "ankle_pitch", "shoulder_yaw", "elbow", "wrist_pitch")
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        name = m.body(b).name
        if b == 0 or m.geom_contype[g] == 0 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        if not any(k in name for k in keep) or name not in kin_body_names:
            continue
        r, half = float(m.geom_size[g, 0]), float(m.geom_size[g, 1])
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, m.geom_quat[g])
        axis = Rg.reshape(3, 3)[:, 2]
        n = max(2, int(np.ceil(2 * half / r)) + 1)
        for s in np.linspace(-half, half, n):
            bodies.append(kin_body_names.index(name)), centers.append(m.geom_pos[g] + s * axis), radii.append(r)
    for side in ("left", "right"):
        # feet: a 4 cm-radius bar from heel to toe along the sole; hands: a 4.5 cm sphere past the wrist
        foot = kin_body_names.index(f"{side}_ankle_roll_link")
        for x in np.linspace(-0.05, 0.12, 6):
            bodies.append(foot), centers.append(np.array([x, 0.0, -0.012])), radii.append(0.035)
        hand = kin_body_names.index(f"{side}_wrist_yaw_link")
        bodies.append(hand), centers.append(np.array([0.08, 0.0, 0.0])), radii.append(0.045)
    return bodies, np.array(centers, np.float32), np.array(radii, np.float32)


class HeadCamera:
    def __init__(self, cfg: HeadCameraCfg, kin, ground, device: str):
        """kin: G1Kinematics; ground: callable (..., 2) world xy -> height (HeightSampler.ground)."""
        self.cfg, self.kin, self.ground, self.device = cfg, kin, ground, device
        self.mount_body = kin.body_names.index(cfg.mount_link)
        self.mount_pos = torch.tensor(cfg.mount_pos, device=device)
        self.mount_R = axis_angle_to_mat(torch.tensor([0.0, 1.0, 0.0], device=device), torch.tensor(cfg.mount_pitch, device=device))
        v, u = torch.meshgrid(torch.arange(cfg.height, device=device), torch.arange(cfg.width, device=device), indexing="ij")
        self.u, self.v = (u.float() + 0.5).flatten(), (v.float() + 0.5).flatten()  # pixel centers
        b, c, r = _occluder_spheres(kin.body_names)
        self.occ_body = torch.tensor(b, device=device)
        self.occ_center = torch.tensor(c, device=device)
        self.occ_radius = torch.tensor(r, device=device)
        self.shape = (cfg.height, cfg.width)
        # fused kernels for the ray march and the occluder test (~8x faster than eager); static shapes (callers
        # render fixed-size batches; dynamic-shape kernels were 3-4x slower)
        cuda = torch.device(device).type == "cuda"
        self.march = torch.compile(self._march, dynamic=False) if cuda else self._march
        self.occlude = torch.compile(self._occlude, dynamic=False) if cuda else self._occlude

    # ------------------------------------------------------------------ randomization
    def sample_params(self, n: int, noise: bool = True) -> torch.Tensor:
        c, dev = self.cfg, self.device
        p = torch.zeros(n, CAM_PARAM_DIM, device=dev)
        p[:, 9] = torch.randint(0, 5, (n,), device=dev).float()
        if not noise:
            return p
        p[:, 10] = torch.randint(c.latency_steps[0], c.latency_steps[1] + 1, (n,), device=dev).float()
        u = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)  # noqa: E731
        for i in range(3):
            p[:, i] = u(-c.pos_range[i], c.pos_range[i])
        p[:, 3], p[:, 4], p[:, 5] = u(-c.roll_range, c.roll_range), u(-c.pitch_range, c.pitch_range), u(-c.yaw_range, c.yaw_range)
        p[:, 6] = u(-c.fov_range_deg, c.fov_range_deg)
        p[:, 7], p[:, 8] = u(-c.center_range_px, c.center_range_px), u(-c.center_range_px, c.center_range_px)
        return p

    # ------------------------------------------------------------------ rendering
    def camera_pose(self, S, params: torch.Tensor, bodies=None):
        """World origin and rotation of the optical frame (+x forward, +y left, +z up) from the robot state."""
        p, R, v, w, q, qd = S
        bp, bR = bodies if bodies is not None else self.kin.forward(p, R, v, w, q, qd, all_bodies=True)[:2]
        Pt, Rt = bp[:, self.mount_body], bR[:, self.mount_body]
        ex = torch.eye(3, device=self.device)
        Rp = (axis_angle_to_mat(ex[0], params[:, 3]) @ axis_angle_to_mat(ex[1], params[:, 4])
              @ axis_angle_to_mat(ex[2], params[:, 5]))  # fmt: skip
        origin = Pt + (Rt @ (self.mount_pos + params[:, :3]).unsqueeze(-1)).squeeze(-1)
        return origin, Rt @ self.mount_R @ Rp

    def _rays(self, params: torch.Tensor) -> torch.Tensor:
        """(B, P, 3) camera-frame rays with unit forward component."""
        c = self.cfg
        fx = (c.width / 2) / torch.tan(torch.deg2rad(c.hfov_deg + params[:, 6]) / 2)
        fy = (c.height / 2) / torch.tan(torch.deg2rad(c.vfov_deg + params[:, 6] * c.vfov_deg / c.hfov_deg) / 2)
        cx, cy = c.width / 2 + params[:, 7], c.height / 2 + params[:, 8]
        y = -(self.u[None] - cx[:, None]) / fx[:, None]
        z = -(self.v[None] - cy[:, None]) / fy[:, None]
        return torch.stack((torch.ones_like(y), y, z), -1)

    def _march(self, origin, dirs):
        """Image-plane depth of the first ground hit along origin + t * dirs, max_depth + step if none."""
        c = self.cfg
        n = int(math.ceil((c.max_depth + c.step - c.near) / c.step))
        hit = torch.full(dirs.shape[:-1], c.max_depth + c.step, device=dirs.device)
        found = torch.zeros_like(hit, dtype=torch.bool)
        for k in range(1, n + 1):
            t = c.near + k * c.step
            pt = origin[:, None] + t * dirs
            below = pt[..., 2] <= self.ground(pt[..., :2])
            hit = torch.where(below & ~found, torch.full_like(hit, t), hit)
            found = found | below
        lo, hi = (hit - c.step).clamp(min=c.near), hit
        for _ in range(c.refine):
            mid = 0.5 * (lo + hi)
            pt = origin[:, None] + mid[..., None] * dirs
            below = pt[..., 2] <= self.ground(pt[..., :2])
            lo, hi = torch.where(below, lo, mid), torch.where(below, mid, hi)
        return torch.where(found, hi, hit)

    def _occlude(self, depth, origin, dirs, bp, bR):
        centers = bp[:, self.occ_body] + (bR[:, self.occ_body] @ self.occ_center.unsqueeze(-1)).squeeze(-1)  # (B, S, 3)
        a = (dirs * dirs).sum(-1)  # (B, P)
        for s in range(centers.shape[1]):
            oc = origin - centers[:, s]  # (B, 3)
            b = 2 * (dirs * oc[:, None]).sum(-1)
            cc = (oc * oc).sum(-1, keepdim=True) - self.occ_radius[s] ** 2
            disc = b * b - 4 * a * cc
            t = (-b - disc.clamp(min=0).sqrt()) / (2 * a)
            depth = torch.where((disc > 0) & (t > 0) & (t < depth), t, depth)
        return depth

    @torch.no_grad()
    def render(self, S, params: torch.Tensor, noise: bool = True) -> torch.Tensor:
        """Normalized (B, 36, 64) image in [-1, 1] as the policy sees it."""
        c = self.cfg
        p, R, v, w, q, qd = S
        bp, bR = self.kin.forward(p, R, v, w, q, qd, all_bodies=True)[:2]
        origin, Rc = self.camera_pose(S, params, (bp, bR))
        dirs = self._rays(params) @ Rc.transpose(-1, -2)  # (B, P, 3)
        d = self.march(origin, dirs)
        if c.self_occlusion:
            d = self.occlude(d, origin, dirs, bp, bR)
        d = d.clamp(max=c.max_depth)  # misses and far hits read max range ("max" clipping)
        edge = None
        if noise and c.edge_dropout > 0:
            g = d.view(-1, c.height, c.width)
            jump = torch.zeros_like(g)
            jump[:, :, :-1] = (g[:, :, 1:] - g[:, :, :-1]).abs()
            jump[:, :-1, :] = torch.maximum(jump[:, :-1, :], (g[:, 1:, :] - g[:, :-1, :]).abs())
            edge = ((jump > c.edge_jump) & (torch.rand_like(g) < c.edge_dropout)).view_as(d)
        if noise:
            d = d + torch.randn_like(d) * (c.noise_std + c.noise_std_per_m * d)
        invalid = d < c.invalid_below
        if edge is not None:
            invalid = invalid | edge
        x = (d.clamp(0, c.max_depth) / c.max_depth).masked_fill(invalid, -1.0)
        if noise and c.dropout > 0:
            x = x.masked_fill(torch.rand_like(x) < c.dropout, -1.0)
        img = x.view(-1, 1, c.height, c.width)
        t, b, l, r = c.crop
        img = img[:, :, t : c.height - b, l : c.width - r]
        return F.interpolate(img, size=(c.height, c.width), mode="bilinear", align_corners=False)[:, 0]


# uint8 storage of normalized images ([-1, 1] -> 0..255) for the replay buffer
def encode_u8(img: torch.Tensor) -> torch.Tensor:
    return ((img.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


def decode_u8(code: torch.Tensor) -> torch.Tensor:
    return code.float() / 127.5 - 1
