"""Delayed onboard elevation-map approximation, independent of privileged scans."""
from dataclasses import dataclass
import math

import torch


@dataclass
class ElevationCfg:
    enabled: bool = False
    noise: bool = True
    size: tuple[float, float] = (2.0, 1.2)
    resolution: float = .08
    forward_offset: float = .5
    cell_noise: float = .02  # episode sigma U[0.5, 1.5] times this value
    drift: float = .05
    drift_rate: tuple[float, float] = (.03, .15)  # radians/s of bounded sinusoid
    xy_error: float = .05
    yaw_error_degrees: float = 3.
    latency_seconds: float = .10
    refresh_seconds: tuple[float, float] = (.08, .12)
    dropout: tuple[float, float] = (.05, .10)
    blind_radii: tuple[float, float] = (.30, .22)
    sensor_height: float = .65  # above base, approximately head height
    occlusion_samples: int = 16
    occlusion_margin: float = .02
    neutral: float = 0.

    @property
    def shape(self):
        return tuple(round(s / self.resolution) for s in self.size)

    def validate(self):
        values = [*self.size, self.resolution, self.forward_offset, self.cell_noise, self.drift,
                  *self.drift_rate, self.xy_error, self.yaw_error_degrees, self.latency_seconds,
                  *self.refresh_seconds, *self.dropout, *self.blind_radii, self.sensor_height,
                  self.occlusion_margin, self.neutral]
        if not all(math.isfinite(v) for v in values) or self.resolution <= 0 or min(self.size) <= 0:
            raise ValueError("Map geometry and noise settings must be finite, with positive size/resolution")
        if any(n < 1 or abs(n * self.resolution - s) > 1e-6 for n, s in zip(self.shape, self.size)):
            raise ValueError("Map size must be an integer multiple of resolution")
        if min(self.cell_noise, self.drift, self.xy_error, self.yaw_error_degrees, self.latency_seconds,
               self.occlusion_margin, *self.drift_rate) < 0 or min(self.blind_radii) <= 0 or self.sensor_height <= 0:
            raise ValueError("Noise magnitudes must be nonnegative; blind radii and sensor height positive")
        if not 0 <= self.dropout[0] <= self.dropout[1] <= 1 or not 0 < self.refresh_seconds[0] <= self.refresh_seconds[1]:
            raise ValueError("Invalid dropout or refresh range")
        if self.drift_rate[0] > self.drift_rate[1] or self.occlusion_samples < 1:
            raise ValueError("Invalid drift rate or occlusion sample count")


class ElevationMap:
    """Sample static terrain at delayed poses, then hold acquisition-frame maps.

    Occlusion is a heightfield line-of-sight test from above the delayed base to
    each cell. Intermediate terrain above that ray masks the cell. This models
    step shadows, not full lidar ray tracing, overhangs or dynamic obstacles.
    """

    def __init__(self, cfg, n, device, dt, sampler):
        cfg.validate()
        self.cfg, self.dt, self.sampler, self.device = cfg, dt, sampler, device
        axes = [(torch.arange(k, device=device) + .5) * cfg.resolution - s / 2
                for k, s in zip(cfg.shape, cfg.size)]
        axes[0] += cfg.forward_offset
        self.grid = torch.cartesian_prod(*axes)
        self.cells = len(self.grid)
        self.ring_size = round(cfg.latency_seconds / dt) + 1
        self.poses = torch.zeros(n, self.ring_size, 4, device=device)  # xyz, yaw
        self.value = torch.zeros(n, 2 * self.cells, device=device)
        self.stamp = torch.full((n,), -1, dtype=torch.long, device=device)
        self.next_refresh = self.stamp.clone()
        self.acquisition_step = self.stamp.clone()
        self.refresh_step = self.stamp.clone()
        self.start_step = self.stamp.clone()
        self.latency = self.stamp.clone()
        self.period = self.stamp.clone()
        self.shift = torch.zeros(n, 2, device=device)
        self.angle = torch.zeros(n, device=device)
        self.sigma = self.angle.clone()
        self.phase = self.angle.clone()
        self.rate = self.angle.clone()
        self.drop_rate = self.angle.clone()
        self.blind_scale = self.angle.clone()
        self.sensor_z = self.angle.clone()
        self.dropout_mask = torch.zeros(n, self.cells, dtype=torch.bool, device=device)
        self.blind_mask = self.dropout_mask.clone()
        self.shadow_mask = self.dropout_mask.clone()

    def reset(self, ids):
        n, c = len(ids), self.cfg
        def uniform(lo, hi):
            return torch.empty(n, device=self.device).uniform_(lo, hi)
        self.stamp[ids] = -1
        self.next_refresh[ids] = -1
        self.latency[ids] = torch.randint(self.ring_size, (n,), device=self.device)
        self.period[ids] = (uniform(*c.refresh_seconds) / self.dt).round().long().clamp(min=1)
        self.shift[ids] = torch.empty(n, 2, device=self.device).uniform_(-c.xy_error, c.xy_error)
        self.angle[ids] = uniform(-c.yaw_error_degrees, c.yaw_error_degrees) * math.pi / 180
        self.sigma[ids] = uniform(.5, 1.5) * c.cell_noise
        self.phase[ids] = uniform(-math.pi, math.pi)
        self.rate[ids] = uniform(*c.drift_rate)
        self.drop_rate[ids] = uniform(*c.dropout)
        self.blind_scale[ids] = uniform(.8, 1.2)
        self.sensor_z[ids] = uniform(.9, 1.1) * c.sensor_height

    def observe(self, position, yaw, tick):
        c = self.cfg
        fresh = self.stamp < 0
        self.start_step[fresh] = tick
        pose = torch.cat((position, yaw[:, None]), -1)
        self.poses[fresh] = pose[fresh, None]
        new = self.stamp != tick
        self.poses[new, tick % self.ring_size] = pose[new]
        ids = (new & ((tick >= self.next_refresh) | fresh | (not c.noise))).nonzero().flatten()
        self.stamp[new] = tick
        if not len(ids):
            return self.value.clone()
        delay = self.latency[ids] if c.noise else torch.zeros_like(ids)
        past = self.poses[ids, (tick - delay) % self.ring_size]
        angle = past[:, 3] + (self.angle[ids] if c.noise else 0)
        co, si = angle.cos(), angle.sin()
        rotation = torch.stack((co, -si, si, co), -1).reshape(-1, 2, 2)
        local = self.grid[None] + (self.shift[ids, None] if c.noise else 0)
        xy = past[:, None, :2] + local @ rotation.transpose(-1, -2)
        terrain_z = self.sampler.sample(xy)
        heights = terrain_z - past[:, 2:3]
        invalid = torch.zeros_like(heights, dtype=torch.bool)
        if c.noise:
            offset = c.drift * (self.phase[ids] + tick * self.dt * self.rate[ids]).sin()
            heights += offset[:, None] + torch.randn_like(heights) * self.sigma[ids, None]
            dropout = torch.rand_like(heights) < self.drop_rate[ids, None]
            radii = torch.tensor(c.blind_radii, device=self.device)
            blind = ((self.grid[None] / (radii * self.blind_scale[ids, None])[:, None]).square().sum(-1) < 1)
            # Sample along rays in parallel; exclude the sensor and target endpoints.
            fractions = torch.arange(1, c.occlusion_samples + 1, device=self.device) / (c.occlusion_samples + 1)
            ray_xy = past[:, None, None, :2] + fractions[None, None, :, None] * (xy - past[:, None, :2])[:, :, None]
            sensor_z = past[:, 2:3] + self.sensor_z[ids, None]
            ray_z = sensor_z[:, :, None] + fractions * (terrain_z - sensor_z)[:, :, None]
            shadow = (self.sampler.sample(ray_xy) > ray_z + c.occlusion_margin).any(-1)
            invalid = dropout | blind | shadow
            self.dropout_mask[ids], self.blind_mask[ids], self.shadow_mask[ids] = dropout, blind, shadow
        else:
            self.dropout_mask[ids] = self.blind_mask[ids] = self.shadow_mask[ids] = False
        self.value[ids] = torch.cat((heights.masked_fill(invalid, c.neutral), (~invalid).float()), -1)
        self.refresh_step[ids] = tick
        self.acquisition_step[ids] = torch.maximum(tick - delay, self.start_step[ids])
        self.next_refresh[ids] = tick + self.period[ids]
        return self.value.clone()
