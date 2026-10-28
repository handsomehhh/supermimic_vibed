"""Torch heightfield depth camera; terrain only, robot self-occlusion ignored."""
from dataclasses import dataclass
import math

import torch

from .rotations import axis_angle_to_mat


@dataclass
class DepthCfg:
    enabled: bool = False
    noise: bool = True
    resolution: tuple[int, int] = (32, 24)  # width, height
    horizontal_fov_degrees: float = 87.
    mount_link: str = "torso_link"
    mount_position: tuple[float, float, float] = (.0576235, .01753, .42987)  # G1 d435_link
    pitch_degrees: float = -50.  # optical axis elevation in the link's x/z plane
    near: float = .1
    far: float = 3.
    coarse_step: float = .1
    refinement_steps: int = 6
    refresh_seconds: tuple[float, float] = (1 / 30, 1 / 30)
    latency_seconds: float = .06
    depth_noise: float = .005  # sigma [m] = coefficient * depth [m]^2
    dropout: tuple[float, float] = (.02, .05)
    edge_threshold: float = .10
    edge_dropout: float = .3
    position_error: float = .01
    rotation_error_degrees: float = 1.
    neutral: float = 0.

    @property
    def shape(self):
        return tuple(reversed(self.resolution))

    def validate(self):
        if len(self.resolution) != 2 or any(not isinstance(n, int) or n < 1 for n in self.resolution):
            raise ValueError("Depth resolution must be positive integer (width, height)")
        if len(self.mount_position) != 3 or not self.mount_link:
            raise ValueError("Depth camera requires a mount link and xyz offset")
        values = [self.horizontal_fov_degrees, *self.mount_position, self.pitch_degrees, self.near, self.far,
                  self.coarse_step, *self.refresh_seconds, self.latency_seconds, self.depth_noise, *self.dropout,
                  self.edge_threshold, self.edge_dropout, self.position_error, self.rotation_error_degrees, self.neutral]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Depth camera settings must be finite")
        if not 0 < self.horizontal_fov_degrees < 180 or not 0 < self.near < self.far or self.coarse_step <= 0:
            raise ValueError("Invalid depth FOV, clipping planes or coarse step")
        if not isinstance(self.refinement_steps, int) or self.refinement_steps < 0:
            raise ValueError("refinement_steps must be a nonnegative integer")
        if min(self.latency_seconds, self.depth_noise, self.position_error, self.rotation_error_degrees, self.edge_threshold) < 0:
            raise ValueError("Depth noise and latency must be nonnegative")
        if not 0 <= self.dropout[0] <= self.dropout[1] <= 1 or not 0 <= self.edge_dropout <= 1:
            raise ValueError("Invalid depth dropout probability")
        if not 0 < self.refresh_seconds[0] <= self.refresh_seconds[1]:
            raise ValueError("Invalid depth refresh interval")


class DepthCamera:
    """Pinhole axial depth in meters, row-major top to bottom, left to right.

    Output is all depths then all validity bits; misses/holes use neutral, mask=0.
    Rays use optical forward=link +x, image right=-y, image down=-z before pitch.
    Only the HeightSampler surface is rendered, with no robot self-occlusion.
    Coarse first-hit brackets are bisected; features narrower than a coarse ray
    interval can be missed. This follows the bilinear sampler, not mesh triangles.
    Delayed full link poses include waist motion, roll and pitch. Images remain
    in their acquisition frame until refreshed. Noise=False bypasses timing too,
    matching ElevationMap's clean diagnostic mode.
    """

    def __init__(self, cfg, n, device, dt, sampler):
        cfg.validate()
        self.cfg, self.dt, self.sampler, self.device = cfg, dt, sampler, device
        w, h = cfg.resolution
        focal = w / (2 * math.tan(math.radians(cfg.horizontal_fov_degrees) / 2))
        v, u = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        self.rays = torch.stack((torch.ones_like(u), -(u + .5 - w / 2) / focal,
                                -(v + .5 - h / 2) / focal), -1).float().reshape(-1, 3)
        pitch = math.radians(cfg.pitch_degrees)
        co, si = math.cos(pitch), math.sin(pitch)
        self.mount_rotation = torch.tensor([[co, 0, -si], [0, 1, 0], [si, 0, co]], device=device)
        self.mount_position = torch.tensor(cfg.mount_position, device=device)
        self.cells = w * h
        self.ring_size = round(cfg.latency_seconds / dt) + 1
        self.poses = torch.zeros(n, self.ring_size, 12, device=device)
        self.value = torch.zeros(n, 2 * self.cells, device=device)
        self.stamp = torch.full((n,), -1, dtype=torch.long, device=device)
        self.start_step = self.stamp.clone()
        self.acquisition_step = self.stamp.clone()
        self.refresh_step = self.stamp.clone()
        self.latency = self.stamp.clone()
        self.next_refresh = torch.zeros(n, device=device)
        self.period = self.next_refresh.clone()
        self.drop_rate = self.next_refresh.clone()
        self.shift = torch.zeros(n, 3, device=device)
        self.error_rotation = torch.eye(3, device=device).expand(n, -1, -1).clone()
        # Fuse repeated bilinear lookups on CUDA without allocating a ray x step volume.
        self._render = torch.compile(self._march, dynamic=True) if torch.device(device).type == "cuda" else self._march

    def reset(self, ids):
        n, c = len(ids), self.cfg
        def uniform(lo, hi):
            return torch.empty(n, device=self.device).uniform_(lo, hi)
        self.stamp[ids] = -1
        self.next_refresh[ids] = -1
        self.latency[ids] = torch.randint(self.ring_size, (n,), device=self.device)
        self.period[ids] = (uniform(*c.refresh_seconds) / self.dt).clamp(min=1)
        self.drop_rate[ids] = uniform(*c.dropout)
        self.shift[ids] = torch.empty(n, 3, device=self.device).uniform_(-c.position_error, c.position_error)
        rotation = torch.eye(3, device=self.device).expand(n, -1, -1)
        for axis in torch.eye(3, device=self.device):
            angle = uniform(-c.rotation_error_degrees, c.rotation_error_degrees) * math.pi / 180
            rotation = rotation @ axis_angle_to_mat(axis, angle)
        self.error_rotation[ids] = rotation

    def _march(self, origin, directions):
        c = self.cfg
        near_point = origin[:, None] + c.near * directions
        outside = near_point[..., 2] > self.sampler.sample(near_point[..., :2])
        hi = torch.full_like(outside, c.far, dtype=origin.dtype)
        found = ~outside
        steps = math.ceil((c.far - c.near) / c.coarse_step)
        step = (c.far - c.near) / steps
        for k in range(1, steps + 1):
            distance = c.near + k * step
            point = origin[:, None] + distance * directions
            below = point[..., 2] <= self.sampler.sample(point[..., :2])
            hi = torch.where(below & ~found, distance, hi)
            found = found | below
        lo = (hi - step).clamp(min=c.near)
        for _ in range(c.refinement_steps):
            mid = (lo + hi) * .5
            point = origin[:, None] + mid[..., None] * directions
            below = point[..., 2] <= self.sampler.sample(point[..., :2])
            lo, hi = torch.where(below, lo, mid), torch.where(below, mid, hi)
        valid = outside & found
        return hi.masked_fill(~valid, c.neutral), valid

    def render(self, origin, rotation):
        """Clean depth/mask from optical pose, with +x forward and +z up."""
        return self._render(origin, self.rays[None] @ rotation.transpose(-1, -2))

    def observe(self, position, rotation, tick):
        c = self.cfg
        fresh = self.stamp < 0
        self.start_step[fresh] = tick
        pose = torch.cat((position, rotation.flatten(1)), -1)
        self.poses[fresh] = pose[fresh, None]
        new = self.stamp != tick
        self.poses[new, tick % self.ring_size] = pose[new]
        ids = (new & ((tick >= self.next_refresh) | fresh | (not c.noise))).nonzero().flatten()
        self.stamp[new] = tick
        if not len(ids):
            return self.value.clone()
        delay = self.latency[ids] if c.noise else torch.zeros_like(ids)
        past = self.poses[ids, (tick - delay) % self.ring_size]
        body_rotation = past[:, 3:].reshape(-1, 3, 3)
        offset = self.mount_position + (self.shift[ids] if c.noise else 0)
        origin = past[:, :3] + (body_rotation @ offset[..., None]).squeeze(-1)
        optical_rotation = body_rotation @ (self.error_rotation[ids] if c.noise else torch.eye(3, device=self.device)) @ self.mount_rotation
        depth, valid = self.render(origin, optical_rotation)
        if c.noise:
            image = depth.reshape(-1, *c.shape)
            mask = valid.reshape_as(image)
            edge = torch.zeros_like(mask)
            dx = ((image[:, :, 1:] - image[:, :, :-1]).abs() > c.edge_threshold) | (mask[:, :, 1:] != mask[:, :, :-1])
            dy = ((image[:, 1:] - image[:, :-1]).abs() > c.edge_threshold) | (mask[:, 1:] != mask[:, :-1])
            edge[:, :, 1:] |= dx
            edge[:, :, :-1] |= dx
            edge[:, 1:] |= dy
            edge[:, :-1] |= dy
            valid = valid & (torch.rand_like(depth) >= self.drop_rate[ids, None])
            valid &= ~(edge.flatten(1) & (torch.rand_like(depth) < c.edge_dropout))
            depth = depth + torch.randn_like(depth) * c.depth_noise * depth.square()
            valid &= (depth >= c.near) & (depth <= c.far)
        self.value[ids] = torch.cat((depth.masked_fill(~valid, c.neutral), valid.float()), -1)
        self.refresh_step[ids] = tick
        self.acquisition_step[ids] = torch.maximum(tick - delay, self.start_step[ids])
        self.next_refresh[ids] = torch.where(fresh[ids], tick + self.period[ids], self.next_refresh[ids] + self.period[ids])
        return self.value.clone()
