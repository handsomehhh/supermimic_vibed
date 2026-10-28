"""Privileged terrain velocity teacher: 50 Hz MuJoCo-Warp, rsl-rl TensorDict API."""
from __future__ import annotations

from dataclasses import dataclass, field
import os

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .env import MAX_JOINT_VEL, MAX_ROOT_VEL, use_warp_stream
from .elevation_map import ElevationCfg, ElevationMap
from .robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list, sim_model
from .rotations import quat_to_mat
from .terrain import HeightSampler, TILE_TYPES, TerrainCfg, generate
from .stairs import body_contact_vectors, clearance_cost, nearest_edge_heading, stumble_cost


@dataclass
class LocoCfg:
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    collision: str = "proxy"
    seed: int = 42
    max_init_level: int = 0
    episode_seconds: float = 20.0
    curriculum: bool = True
    randomization: bool = True
    observation_noise: bool = True
    command_seconds: tuple[float, float] = (8.0, 12.0)
    command_x: tuple[float, float] = (-1.0, 1.0)
    command_y: tuple[float, float] = (-1.0, 1.0)
    command_yaw: tuple[float, float] = (-0.5, 0.5)
    standing_fraction: float = 0.1
    heading_fraction: float = 0.3
    forward_fraction: float = 0.2
    command_curriculum: bool = True
    scan_shape: tuple[int, int] = (17, 11)
    scan_size: tuple[float, float] = (1.6, 1.0)
    elevation: ElevationCfg = field(default_factory=ElevationCfg)
    history_length: int = 5
    # Opt-in P2.5 terms: disabled branches consume no random numbers or observations.
    foot_clearance_weight: float = 0.0
    foot_clearance_target: float = 0.10
    foot_stumble_weight: float = 0.0
    foot_stumble_ratio: float = 3.0
    outward_fraction: float = 0.0
    outward_speed: float = 0.6
    contact_vectors: bool = False  # diagnostics can request vectors without rewards


STAIRS_OPTIONS = ("foot_clearance_weight", "foot_clearance_target", "foot_stumble_weight",
                  "foot_stumble_ratio", "outward_fraction", "outward_speed", "contact_vectors")


def setup_device(device: str) -> None:
    # All generated caches remain in the repository, including local GPU tests.
    cache = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
    os.environ.setdefault("CUDA_CACHE_PATH", os.path.join(cache, "cuda"))
    wp.config.kernel_cache_dir = os.path.join(cache, "warp")
    torch.set_num_threads(4)
    use_warp_stream(device)
    torch.backends.cuda.matmul.allow_tf32 = True


@wp.kernel
def _body_contacts(nacon: wp.array[int], world: wp.array[int], geom: wp.array[wp.vec2i],
                   address: wp.array2d[int], force: wp.array2d[float], bodies: wp.array[int],
                   normal: wp.array2d[float]):
    i = wp.tid()
    if i < nacon[0]:
        w = world[i]
        f = float(0.0)
        # condim=3, pyramidal cone: the four pyramid forces sum to normal force.
        for k in range(4):
            a = address[i, k]
            if a >= 0 and a < force.shape[1]:
                f += force[w, a]
        for k in range(2):
            g = geom[i][k]
            if g >= 0:
                b = bodies[g]
                if b > 0:
                    wp.atomic_add(normal, w, b, wp.max(f, 0.0) * 0.25)


def _expand_model(model, name: str, n: int) -> torch.Tensor:
    src = getattr(model, name)
    values = wp.to_torch(src).repeat(n, *([1] * (wp.to_torch(src).ndim - 1))).contiguous()
    dst = wp.from_torch(values, dtype=src.dtype)
    setattr(model, name, dst)
    return wp.to_torch(dst)


class LocoEnv(VecEnv):
    """Call setup_device once before construction; step auto-resets training worlds."""

    num_actions = 29
    dt = 0.02

    def __init__(self, num_envs: int = 4096, cfg: LocoCfg | None = None, device: str = "cuda:0",
                 eval_mode: bool = False):
        self.cfg = cfg or LocoCfg()
        if not np.isfinite([getattr(self.cfg, k) for k in STAIRS_OPTIONS]).all():
            raise ValueError("Stair options must be finite")
        if not 0 <= self.cfg.outward_fraction <= 1:
            raise ValueError("outward_fraction must be in [0, 1]")
        if min(self.cfg.foot_clearance_target, self.cfg.foot_stumble_ratio, self.cfg.outward_speed) <= 0:
            raise ValueError("Clearance target, stumble ratio and outward speed must be positive")
        if max(self.cfg.foot_clearance_weight, self.cfg.foot_stumble_weight) > 0:
            raise ValueError("Clearance and stumble weights must be nonpositive costs")
        self.device, self.num_envs, self.eval_mode = device, num_envs, eval_mode
        self.common_step_counter = 0
        self.max_episode_length = round(self.cfg.episode_seconds / self.dt)
        self.terrain = generate(self.cfg.terrain, self.cfg.seed)
        self.sampler = HeightSampler(self.terrain, device)
        self.grid = torch.cartesian_prod(*[torch.linspace(-s / 2, s / 2, n, device=device)
                                          for s, n in zip(self.cfg.scan_size, self.cfg.scan_shape)])
        self.origins = torch.tensor(self.terrain.origins, device=device)
        self.tile_types = torch.tensor(self.terrain.tile_types, device=device)
        self.ids = torch.arange(num_envs, device=device)
        self.columns = self.ids % self.cfg.terrain.num_cols
        self.levels = torch.randint(min(self.cfg.max_init_level + 1, self.cfg.terrain.num_rows), (num_envs,), device=device)
        self.kinds = self.tile_types[0, self.columns]
        self.type_masks = {name: self.kinds == k for k, name in enumerate(TILE_TYPES) if (self.terrain.tile_types == k).any()}
        self.mjm = m = sim_model(terrain=self.terrain, collision=self.cfg.collision)
        m.opt.ccd_iterations = 50
        m.opt.ccd_tolerance = 1e-4  # FP32 contact witnesses on a 170 m shared hfield
        assert [m.joint(i).name for i in range(1, m.njnt)] == G1_JOINT_NAMES
        self.default_q = torch.tensor(default_joint_pos_list(), device=device)
        self.scale = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=device)
        self.q_lo, self.q_hi = torch.tensor(m.jnt_range[1:].T, device=device, dtype=torch.float32)
        center, half = (self.q_hi + self.q_lo) / 2, (self.q_hi - self.q_lo) * 0.45
        self.soft_lo, self.soft_hi = center - half, center + half
        self.feet = torch.tensor([m.body(s + "_ankle_roll_link").id for s in ("left", "right")], device=device)
        self.nonfeet = torch.ones(m.nbody, device=device, dtype=torch.bool)
        self.nonfeet[0], self.nonfeet[self.feet] = False, False
        self._vector_contacts = self.cfg.contact_vectors or self.cfg.foot_stumble_weight != 0
        if self.cfg.foot_clearance_weight or self.cfg.contact_vectors:
            # Both collision modes use spherical sole pads. Probe their actual lower
            # surfaces, including the toe, instead of treating ankle height as clearance.
            pads = [np.flatnonzero((m.geom_bodyid == b) & (m.geom_contype != 0)
                                  & (m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE)) for b in self.feet.tolist()]
            if not len(pads[0]) or len(pads[0]) != len(pads[1]):
                raise ValueError("Stair clearance requires matching spherical sole pads on both feet")
            self.foot_pad_ids = torch.tensor(np.stack(pads), device=device)
            self.foot_pad_radius = torch.tensor(m.geom_size[np.stack(pads), 0], device=device, dtype=torch.float32)
        self.hips = [i for i, n in enumerate(G1_JOINT_NAMES) if "hip_roll" in n or "hip_yaw" in n]
        self.arms = list(range(15, 29))
        with wp.ScopedDevice(device):
            self.model = mjw.put_model(m)
            self.data = mjw.make_data(m, nworld=num_envs, nconmax=128, nccdmax=32, njmax=1000)
            self._normal = wp.zeros((num_envs, m.nbody), dtype=float)
            if self._vector_contacts:
                self._force_vector = wp.zeros((num_envs, m.nbody, 3), dtype=float)
            self._reset_mask = wp.zeros(num_envs, dtype=bool)
            friction = _expand_model(self.model, "geom_friction", num_envs)
            ipos = _expand_model(self.model, "body_ipos", num_envs)
            mass = _expand_model(self.model, "body_mass", num_envs)
            inertia = _expand_model(self.model, "body_inertia", num_envs)
            # set_const writes these derived arrays, which must also be per-world.
            for name in ("body_subtreemass", "body_invweight0", "dof_invweight0", "dof_M0", "actuator_acc0"):
                if hasattr(self.model, name):
                    _expand_model(self.model, name, num_envs)
            _expand_model(self.model.stat, "meaninertia", num_envs)
        self.qpos, self.qvel, self.ctrl = [wp.to_torch(a) for a in (self.data.qpos, self.data.qvel, self.data.ctrl)]
        self.body_force = wp.to_torch(self._normal)
        if self._vector_contacts:
            self.body_force_vector = wp.to_torch(self._force_vector)
        self.friction = torch.ones(num_envs, device=device)
        if self.cfg.randomization and not eval_mode:
            self.friction.uniform_(0.8, 1.2)
            torso = m.body("torso_link").id
            ipos[:, torso] += torch.empty(num_envs, 3, device=device).uniform_(-1, 1) * torch.tensor([.025, .025, .03], device=device)
            factor = torch.empty(num_envs, device=device).uniform_(0.8, 1.2)
            mass[:, torso] *= factor
            inertia[:, torso] *= factor[:, None]
        friction[:, :, 0] = self.friction[:, None]
        friction[:, m.geom_bodyid == 0, 0] = 0.8 if not eval_mode else 1.0
        with wp.ScopedDevice(device):
            mjw.set_const(self.model, self.data)
        self.stand_height = self._stand_height()
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_action = torch.zeros(num_envs, 29, device=device)
        self.cmd = torch.zeros(num_envs, 3, device=device)
        self.heading = torch.zeros(num_envs, device=device)
        self.heading_mode = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.standing = self.heading_mode.clone()
        if self.cfg.outward_fraction:
            self.outward_mode = self.heading_mode.clone()
            self.outward_velocity = torch.zeros(num_envs, device=device)
        self.command_left = self.episode_length_buf.clone()
        self.push_left = self.episode_length_buf.clone()
        self.start_xy = torch.zeros(num_envs, 2, device=device)
        self.command_distance = torch.zeros(num_envs, device=device)
        self.air_time = torch.zeros(num_envs, 2, device=device)
        self.foot_contact_time = self.air_time.clone()
        self.ep_return = torch.zeros(num_envs, device=device)
        self.total_blown = torch.zeros((), device=device, dtype=torch.long)
        self.curriculum_up = self.total_blown.clone()
        self.curriculum_down = self.total_blown.clone()
        self.extras = {}
        self.elevation = None
        if self.cfg.elevation.enabled:
            if self.cfg.history_length < 1:
                raise ValueError("history_length must be positive")
            self.elevation = ElevationMap(self.cfg.elevation, num_envs, device, self.dt, self.sampler)
            self.policy_history = torch.zeros(num_envs, self.cfg.history_length, 96, device=device)
            self.history_stamp = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.reset(self.ids)
        with wp.ScopedDevice(device), wp.ScopedCapture() as cap:
            self._normal.zero_()
            if self._vector_contacts:
                self._force_vector.zero_()
            for _ in range(4):
                mjw.step(self.model, self.data)
                c = self.data.contact
                wp.launch(_body_contacts, dim=self.data.naconmax,
                          inputs=[self.data.nacon, c.worldid, c.geom, c.efc_address, self.data.efc.force,
                                  self.model.geom_bodyid, self._normal])
                if self._vector_contacts:
                    wp.launch(body_contact_vectors, dim=self.data.naconmax,
                              inputs=[self.data.nacon, c.worldid, c.geom, c.efc_address, self.data.efc.force,
                                      self.model.geom_bodyid, c.frame, c.friction, self._force_vector])
            # mjw.step leaves kinematics at the beginning of the final substep.
            mjw.kinematics(self.model, self.data)
        self.step_graph = cap.graph
        self.reset(self.ids)

    def _stand_height(self) -> float:
        d = mujoco.MjData(self.mjm)
        d.qpos[3], d.qpos[7:] = 1, default_joint_pos_list()
        mujoco.mj_kinematics(self.mjm, d)
        low = []
        for g in range(self.mjm.ngeom):
            if self.mjm.geom_bodyid[g] not in self.feet.tolist() or not self.mjm.geom_contype[g]:
                continue
            size = self.mjm.geom_size[g]
            extent = size[0] if self.mjm.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE else abs(d.geom_xmat[g].reshape(3, 3)[2]) @ size
            low.append(d.geom_xpos[g, 2] - extent)
        return float(-min(low) + .002)

    def resample_commands(self, ids: torch.Tensor) -> None:
        n = len(ids)
        if not n:
            return
        for k, limits in enumerate((self.cfg.command_x, self.cfg.command_y, self.cfg.command_yaw)):
            self.cmd[ids, k] = torch.empty(n, device=self.device).uniform_(*limits)
        if self.cfg.command_curriculum:
            scale = .5 if self.common_step_counter < 3000 * 24 else (.7 if self.common_step_counter < 8000 * 24 else 1.0)
            self.cmd[ids, :2] *= scale
        self.heading[ids] = torch.empty(n, device=self.device).uniform_(-np.pi, np.pi)
        self.heading_mode[ids] = torch.rand(n, device=self.device) < self.cfg.heading_fraction
        forward = ids[torch.rand(n, device=self.device) < self.cfg.forward_fraction]
        self.cmd[forward, 0] = self.cmd[forward, 0].abs().clamp(min=.3)
        self.cmd[forward, 1:] = 0
        self.heading_mode[forward] = False
        self.standing[ids] = torch.rand(n, device=self.device) < self.cfg.standing_fraction
        self.command_left[ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.command_seconds).div(self.dt).long()
        if self.cfg.outward_fraction:
            pyramid = (self.kinds[ids] >= 2) & (self.kinds[ids] <= 5)
            self.outward_mode[ids] = pyramid & (torch.rand(n, device=self.device) < self.cfg.outward_fraction)
            selected = ids[self.outward_mode[ids]]
            R = quat_to_mat(self.qpos[selected, 3:7])
            yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
            self.heading[selected] = nearest_edge_heading(
                self.qpos[selected, :2], self.origins[self.levels[selected], self.columns[selected], :2],
                self.qpos.new_tensor(self.cfg.terrain.tile_size) / 2, yaw)
            self.heading_mode[selected], self.standing[selected] = True, False
            self.outward_velocity[selected] = self.cfg.outward_speed * (scale if self.cfg.command_curriculum else 1.)
        self._heading_commands()

    def _heading_commands(self) -> None:
        R = quat_to_mat(self.qpos[:, 3:7])
        error = self.heading - torch.atan2(R[:, 1, 0], R[:, 0, 0])
        rate = (0.5 * torch.atan2(error.sin(), error.cos())).clamp(*self.cfg.command_yaw)
        self.cmd[:, 2] = torch.where(self.heading_mode, rate, self.cmd[:, 2])
        if self.cfg.outward_fraction:
            # Turn toward the edge before translating; body-frame vy stays zero.
            self.cmd[self.outward_mode, 0] = (self.outward_velocity * error.cos().clamp(min=0))[self.outward_mode]
            self.cmd[self.outward_mode, 1] = 0
        self.cmd[self.standing] = 0

    def update_curriculum(self, ids: torch.Tensor) -> None:
        distance = (self.qpos[ids, :2] - self.start_xy[ids]).norm(dim=-1)
        up = distance > self.cfg.terrain.tile_size[0] / 2
        down = (distance < self.command_distance[ids] * 0.5) & ~up
        self.curriculum_up += up.sum()
        self.curriculum_down += down.sum()
        level = (self.levels[ids] + up.long() - down.long()).clamp(min=0)
        random = torch.randint(self.cfg.terrain.num_rows, (len(ids),), device=self.device)
        self.levels[ids] = torch.where(level >= self.cfg.terrain.num_rows, random, level)

    def reset(self, ids: torch.Tensor) -> None:
        if not len(ids):
            return
        mask = wp.to_torch(self._reset_mask)
        mask.zero_()
        mask[ids] = True
        with wp.ScopedDevice(self.device):
            mjw.reset_data(self.model, self.data, reset=self._reset_mask)
        self.qpos[ids] = 0
        self.qpos[ids, :3] = self.origins[self.levels[ids], self.columns[ids]]
        self.qpos[ids, 2] += self.stand_height
        yaw = torch.zeros(len(ids), device=self.device) if self.eval_mode else torch.empty(len(ids), device=self.device).uniform_(-np.pi, np.pi)
        self.qpos[ids, 3], self.qpos[ids, 6] = (yaw / 2).cos(), (yaw / 2).sin()
        self.qpos[ids, 7:] = self.default_q
        self.qvel[ids], self.ctrl[ids] = 0, self.default_q
        self.last_action[ids], self.air_time[ids], self.foot_contact_time[ids] = 0, 0, 0
        self.episode_length_buf[ids], self.ep_return[ids], self.command_distance[ids] = 0, 0, 0
        self.start_xy[ids] = self.qpos[ids, :2]
        self.push_left[ids] = torch.randint(250, 750, (len(ids),), device=self.device)
        if not self.eval_mode:
            self.resample_commands(ids)
            if self.cfg.outward_fraction:
                selected = ids[self.outward_mode[ids]]
                angle = self.heading[selected] / 2
                self.qpos[selected, 3], self.qpos[selected, 6] = angle.cos(), angle.sin()
                self._heading_commands()
        if self.elevation is not None:
            self.elevation.reset(ids)
            self.history_stamp[ids] = -1

    def _state(self):
        R = quat_to_mat(self.qpos[:, 3:7])
        vel = (R.transpose(1, 2) @ self.qvel[:, :3, None]).squeeze(-1)
        gravity = -R[:, 2, :]
        height = self.qpos[:, 2] - self.sampler.sample(self.qpos[:, :2])
        return R, vel, gravity, height

    def foot_clearance(self):
        """World sole-pad bottoms, terrain under each pad, and minimum per-foot gap."""
        points = wp.to_torch(self.data.geom_xpos)[:, self.foot_pad_ids].clone()
        points[..., 2] -= self.foot_pad_radius
        terrain = self.sampler.sample(points[..., :2])
        return points, terrain, (points[..., 2] - terrain).amin(-1)

    def get_observations(self) -> TensorDict:
        R, vel, gravity, _ = self._state()
        clean = torch.cat((self.qvel[:, 3:6], gravity, self.cmd, self.qpos[:, 7:] - self.default_q,
                           self.qvel[:, 6:], self.last_action), -1)
        policy = clean.clone()
        if self.cfg.observation_noise and not self.eval_mode:
            noise = torch.empty_like(policy).uniform_(-1, 1)
            policy[:, :3] += noise[:, :3] * .2
            policy[:, 3:6] += noise[:, 3:6] * .05
            policy[:, 9:38] += noise[:, 9:38] * .01
            policy[:, 38:67] += noise[:, 38:67] * 1.5
        scan = (self.sampler.scan(self.qpos[:, :3], R, self.grid) - self.stand_height).clamp(-1, 1)
        groups = {"policy": policy, "height_scan": scan,
                  "critic": torch.cat((clean, scan, vel, self.friction[:, None]), -1)}
        if self.elevation is not None:
            fresh = self.history_stamp < 0
            advance = (self.history_stamp != self.common_step_counter) & ~fresh
            self.policy_history[fresh] = policy[fresh, None]
            self.policy_history[advance] = torch.cat((self.policy_history[advance, 1:], policy[advance, None]), 1)
            self.history_stamp[:] = self.common_step_counter
            groups["policy_history"] = self.policy_history.flatten(1).clone()
            # Keep repeated reads and the latest history frame consistent.
            groups["policy"] = self.policy_history[:, -1].clone()
            groups["elevation_map"] = self.elevation.observe(
                self.qpos[:, :3], torch.atan2(R[:, 1, 0], R[:, 0, 0]), self.common_step_counter)
        return TensorDict(groups, batch_size=[self.num_envs], device=self.device)

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        self.common_step_counter += 1
        bad_action = ~torch.isfinite(actions).all(-1)
        actions = actions.nan_to_num().clamp(-10, 10)
        q0, v0 = self.qpos.clone(), self.qvel.clone()
        self.ctrl[:] = (self.default_q + self.scale * actions).clamp(self.q_lo, self.q_hi)
        wp.capture_launch(self.step_graph)
        bad = bad_action | ~torch.isfinite(self.qpos).all(-1) | ~torch.isfinite(self.qvel).all(-1)
        bad |= (self.qvel[:, :6].abs().amax(-1) > MAX_ROOT_VEL) | (self.qvel[:, 6:].abs().amax(-1) > MAX_JOINT_VEL)
        self.qpos.copy_(torch.where(bad[:, None], q0, self.qpos))
        self.qvel.copy_(torch.where(bad[:, None], v0, self.qvel))
        self.total_blown += bad.sum()
        R, vel, gravity, height = self._state()
        cvel = wp.to_torch(self.data.cvel)[:, self.feet]
        offset = wp.to_torch(self.data.xpos)[:, self.feet] - wp.to_torch(self.data.subtree_com)[:, 1:2]
        foot_speed = cvel[:, :, 3:] + torch.cross(cvel[:, :, :3], offset, dim=-1)
        contact = self.body_force[:, self.feet] > 1.0
        self.air_time = torch.where(contact, 0, self.air_time + self.dt)
        self.foot_contact_time = torch.where(contact, self.foot_contact_time + self.dt, 0)
        # Biped single-support reward: no reward for jumping with both feet airborne.
        support_time = torch.where(contact, self.foot_contact_time, self.air_time).amin(-1).clamp(max=.5)
        moving = self.cmd.norm(dim=-1) > .05
        lin_error = (vel[:, :2] - self.cmd[:, :2]).square().sum(-1)
        yaw_error = (self.qvel[:, 5] - self.cmd[:, 2]).square()
        self.episode_length_buf += 1
        self.command_distance += self.cmd[:, :2].norm(dim=-1) * self.dt
        fall = (height < .35) | (gravity[:, 2] > -np.cos(np.deg2rad(70))) | bad
        timeout = (self.episode_length_buf >= self.max_episode_length) & ~fall
        done = fall | timeout
        dq = self.qpos[:, 7:] - self.default_q
        torque = wp.to_torch(self.data.actuator_force)
        terms = {
            "track_lin": 2 * torch.exp(-lin_error / .25),
            "track_yaw": 2 * torch.exp(-yaw_error / .25),
            "orientation": torch.exp(-gravity[:, :2].square().sum(-1) / .2),
            "base_height": torch.exp(-(height - self.stand_height).square() / .01),
            "vel_z": -2 * vel[:, 2].square(),
            "ang_xy": -.05 * self.qvel[:, 3:5].square().sum(-1),
            "torques": -1e-5 * torque.square().sum(-1),
            "joint_acc": -2.5e-7 * ((self.qvel[:, 6:] - v0[:, 6:]) / self.dt).square().sum(-1),
            "action_rate": -.03 * (actions - self.last_action).square().sum(-1),
            "air_time": .4 * support_time * (contact.sum(-1) == 1) * moving,
            "foot_slide": -.1 * (foot_speed[:, :, :2].square().sum(-1) * contact).sum(-1),
            "body_contact": -1.0 * (self.body_force[:, self.nonfeet] > 10).float().sum(-1),
            "joint_limits": -((self.soft_lo - self.qpos[:, 7:]).clamp(min=0) + (self.qpos[:, 7:] - self.soft_hi).clamp(min=0)).sum(-1),
            "hip_deviation": -.1 * dq[:, self.hips].abs().sum(-1),
            "arm_deviation": -.1 * dq[:, self.arms].abs().sum(-1),
            "waist_deviation": -.1 * dq[:, 12:15].abs().sum(-1),
            "termination": -10.0 * fall.float(),
        }
        if self.cfg.foot_clearance_weight:
            _, _, clearance = self.foot_clearance()
            terms["foot_clearance"] = self.cfg.foot_clearance_weight * clearance_cost(
                clearance, foot_speed[..., :2], contact, self.cmd, self.cfg.foot_clearance_target)
        if self.cfg.foot_stumble_weight:
            terms["foot_stumble"] = self.cfg.foot_stumble_weight * stumble_cost(
                self.body_force_vector[:, self.feet], self.cfg.foot_stumble_ratio)
        terms = {k: torch.where(bad, 0, v) for k, v in terms.items()}
        reward = sum(terms.values()) * self.dt
        self.ep_return += reward
        self.last_action.copy_(actions)
        logs = {f"reward/{k}": v.mean() for k, v in terms.items()}
        logs.update(tracking_lin=lin_error.sqrt().mean(), tracking_yaw=yaw_error.sqrt().mean(),
                    terrain_level=self.levels.float().mean(), blowups=bad.sum(),
                    **{f"level/{k}": self.levels[v].float().sum() / v.sum().clamp(min=1) for k, v in self.type_masks.items()})
        extras = {"log": logs, "time_outs": timeout, "valid": ~bad, "falls": fall,
                  "tracking_lin": lin_error.sqrt(), "tracking_yaw": yaw_error.sqrt()}
        # Preserve terminal observations for correct timeout bootstrap before auto-reset.
        if bool(timeout.any()):
            extras["terminal_observation"] = self.get_observations()
        ids = done.nonzero().flatten()
        extras["episode_returns"] = self.ep_return[ids].clone()
        extras["episode_lengths"] = self.episode_length_buf[ids].clone()
        if not self.eval_mode:
            if self.cfg.curriculum:
                self.update_curriculum(ids[~bad[ids]])
            self.reset(ids)
            self.command_left -= 1
            self.resample_commands((self.command_left <= 0).nonzero().flatten())
            self._heading_commands()
            if self.cfg.randomization:
                self.push_left -= 1
                push = (self.push_left <= 0).nonzero().flatten()
                self.qvel[push, :2] += torch.empty(len(push), 2, device=self.device).uniform_(-.5, .5)
                self.push_left[push] = torch.randint(250, 750, (len(push),), device=self.device)
        elif len(ids[bad[ids]]):
            self.reset(ids[bad[ids]])
        self.extras = extras
        return self.get_observations(), reward, done, extras
