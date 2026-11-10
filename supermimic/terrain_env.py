"""TrackingSim with the velocity interface on terrain and a head depth camera (motion tracking stays on the floor).

Scene: the z = 0 floor plane (motion-tracking worlds, which follow their clips around the origin) plus a
heightfield map placed away from the origin (velocity worlds). The map is mjlab's rough-terrain generator
without random-rough tiles (terrain.mjlab_rough_cfg). Collision geometry is hybrid (robot.sim_model): the mesh
geoms collide with the floor and with each other exactly as in the flat runs; primitive proxy copies collide with
the heightfield only (mesh-heightfield contacts blow up, PERCEPTION_MILESTONES P2; proxy self-collisions
overlap at the hips in the default pose).

Velocity worlds: one tile per world (column = world index mod columns, level from the curriculum), spawned on
the tile's center platform from a velocity-library state (yaw 0 on stairs, uniform elsewhere), 20 s episodes.
Commands follow mjlab's UniformVelocityCommand (resampled every 3-8 s; 10% standing, 30% heading control
with stiffness 0.5, 20% forward; forward-only with heading 0 on stairs) at its final curriculum ranges
vx (-0.8, 1.0), vy (-0.7, 0.7), yaw (-0.7, 0.7); flat tiles keep the flat-ground interface's commands
(reference commands and the full ranges). Terrain-level curriculum as mjlab terrain_levels_vel.
Reward, fall and timeout: terrain_mdp.TerrainTask; out of the map's interior ends the episode (timeout).
Depth: head_camera.HeadCamera at 30 Hz, a 10-frame stack per velocity world.
"""

from __future__ import annotations

from dataclasses import replace
import math

import torch

from .env import TrackingSim
from .head_camera import CAM_PARAM_DIM, HeadCamera, arrives, encode_u8, max_latency, ring_len
from .robot import sim_model
from .rotations import axis_angle_to_mat
from .terrain import TILE_TYPES, HeightSampler, generate
from .terrain_mdp import STAIRS, TerrainTask
from .terrain_setup import TerrainSetup

FLAT_KIND, STAIR_KINDS = TILE_TYPES.index("flat"), (TILE_TYPES.index("stairs_up"), TILE_TYPES.index("stairs_down"))


class TerrainSim(TrackingSim):
    def __init__(self, *args, setup: TerrainSetup | None = None, **kw):
        self.setup = setup or TerrainSetup()
        super().__init__(*args, **kw)

    # ------------------------------------------------------------------ construction hooks
    def _build_model(self, timestep: float, self_collision: bool):
        st = self.setup
        self.map = generate(st.terrain, st.seed)
        self.hs = HeightSampler(self.map, self.device)
        self.hs_cam = self.hs
        if st.camera_scale and st.camera_scale != st.terrain.horizontal_scale:
            self.hs_cam = HeightSampler(generate(replace(st.terrain, horizontal_scale=st.camera_scale), st.seed), self.device)
        m = sim_model(timestep, terrain=self.map, collision="hybrid", self_collision=self_collision, keep_floor=True)
        m.opt.ccd_iterations = 50
        m.opt.ccd_tolerance = 1e-4  # FP32 contact witnesses far from the origin
        return m

    def _data_sizes(self) -> dict:
        return dict(nconmax=128, nccdmax=32, njmax=1000)

    def _init_extra(self):
        st, dev, N = self.setup, self.device, self.N
        self.task = TerrainTask(self.mdp.kin, self.hs.ground, self.mdp, dev, st.reward)
        self.cam = HeadCamera(st.camera, self.mdp.kin, self.hs_cam.ground, dev)
        self.vel_lo = N - self.n_vel
        self.origins = torch.tensor(self.map.origins, device=dev)  # (rows=levels, cols, 3)
        self.kinds_map = torch.tensor(self.map.tile_types, device=dev)
        nr, nc = self.origins.shape[:2]
        self.col = torch.arange(N, device=dev) % nc
        self.level = torch.randint(0, min(st.max_init_level, nr - 1) + 1, (N,), device=dev)
        self.kind = self.kinds_map[0, self.col]
        self.info = torch.zeros(N, 2, device=dev)
        self._update_info()
        self.start_xy = torch.zeros(N, 2, device=dev)
        self.heading_mode = torch.zeros(N, dtype=torch.bool, device=dev)
        self.standing = torch.zeros(N, dtype=torch.bool, device=dev)
        self.heading_target = torch.zeros(N, device=dev)
        self.cmd_timer = torch.zeros(N, dtype=torch.long, device=dev)
        self.cam_params = self.cam.sample_params(N, noise=not self.eval_mode)
        # arrived frames, oldest first (the policy sees every `stride`-th one, see depth_stack), and the frames
        # rendered at the last max_latency + 1 control steps (newest last), from which delayed frames arrive
        self.depth_ring = torch.zeros(self.n_vel, ring_len(st.camera), *self.cam.shape, device=dev, dtype=torch.float16)
        self.render_hist = torch.zeros(self.n_vel, max_latency(st.camera) + 1, *self.cam.shape, device=dev, dtype=torch.float16)
        self.tick = torch.ones(N, dtype=torch.bool, device=dev)
        self.fresh = torch.zeros(N, dtype=torch.bool, device=dev)  # velocity worlds reset since their last frame
        self._in_step = False
        lo = torch.tensor(self.map.offset, device=dev)
        self.map_lo = lo + st.bounds_margin
        self.map_hi = lo + torch.tensor(self.map.extent, device=dev) - st.bounds_margin
        self.reset_count = 0

    def _update_info(self):
        stairs = (self.kind == STAIR_KINDS[0]) | (self.kind == STAIR_KINDS[1])
        self.info[:, 0] = torch.where(stairs, STAIRS, 0.0)
        self.info[:, 1] = self.level.float()

    @property
    def vel_max_step_reward(self) -> float:
        return self.task.max_step_reward

    # ------------------------------------------------------------------ resets
    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        super().reset(env_ids, frames)
        if not hasattr(self, "task"):
            return
        vel = env_ids[self.is_vel[env_ids]]
        if len(vel):
            self._place_on_tiles(vel)
            self._camera_reset(vel)

    def reset_standing(self, env_ids: torch.Tensor, frames: torch.Tensor):
        super().reset_standing(env_ids, frames)
        self._camera_reset(env_ids[self.is_vel[env_ids]])

    def reset_on_tile(self, env_ids: torch.Tensor, col: torch.Tensor, level: torch.Tensor, yaw: torch.Tensor, frames: torch.Tensor):
        """Default standing pose at the center of tile (level, col) with the given yaw (terrain evaluation)."""
        self.col[env_ids], self.level[env_ids] = col, level
        self.kind[env_ids] = self.kinds_map[0, col]
        self._update_info()
        super().reset_standing(env_ids, frames)
        o = self.origins[level, col]
        p = self.qpos[env_ids, :3].clone()
        p[:, :2] = o[:, :2]
        p[:, 2] = self.stand_height + self.hs.ground(o[:, :2])
        R = axis_angle_to_mat(torch.tensor([0.0, 0.0, 1.0], device=self.device), yaw)
        zero = torch.zeros(len(env_ids), 3, device=self.device)
        self.write_state(env_ids, p, R, zero, zero, self.qpos[env_ids, 7:].clone(), torch.zeros(len(env_ids), 29, device=self.device))
        self.start_xy[env_ids] = p[:, :2]
        self._camera_reset(env_ids)

    def _place_on_tiles(self, ids: torch.Tensor):
        st, dev, n = self.setup, self.device, len(ids)
        p, R, v, w, q, qd = (x[ids] for x in self.state())
        yaw_now = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        stairs = self.info[ids, 0] > 0.5
        yaw = torch.where(stairs, 0.0, torch.empty(n, device=dev).uniform_(-math.pi, math.pi))
        Rz = axis_angle_to_mat(torch.tensor([0.0, 0.0, 1.0], device=dev), yaw - yaw_now)
        o = self.origins[self.level[ids], self.col[ids]]
        xy = o[:, :2] + torch.empty(n, 2, device=dev).uniform_(-st.spawn_xy, st.spawn_xy)
        p = torch.cat((xy, (p[:, 2] + self.hs.ground(xy)).unsqueeze(-1)), -1)  # clip heights are above a flat floor
        rot = lambda x: (Rz @ x.unsqueeze(-1)).squeeze(-1)  # noqa: E731
        self.write_state(ids, p, Rz @ R, rot(v), rot(w), q, qd)
        self.start_xy[ids] = xy

    def _camera_reset(self, ids: torch.Tensor):
        """New camera parameters; the first frame (filling the whole stack) comes with the next frame update."""
        if not len(ids):
            return
        self.cam_params[ids] = self.cam.sample_params(len(ids), noise=not self.eval_mode)
        self.fresh[ids] = True
        self.tick[ids] = True
        if not self._in_step:  # resets outside step() (construction, evaluation) get their frame right away
            self._update_frames(torch.zeros(self.N, dtype=torch.bool, device=self.device))

    @property
    def depth_stack(self) -> torch.Tensor:
        """(n_vel, frames, H, W) policy depth input, oldest first."""
        return self.depth_ring[:, :: self.setup.camera.stride]

    def _update_frames(self, tick: torch.Tensor):
        """One fixed-size render of all velocity worlds (a fixed shape keeps the compiled renderer fast); worlds
        reset since their last frame fill their ring (and render history) with it; worlds where a frame arrives
        (``tick``) push the frame rendered `latency` steps ago."""
        if not self.n_vel:
            return
        v = slice(self.vel_lo, self.N)
        S = tuple(x[v] for x in self.state())
        frame = self.cam.render(S, self.cam_params[v], noise=not self.eval_mode).half()
        fresh, tick = self.fresh[v], tick[v] & ~self.fresh[v]
        hist = torch.cat((self.render_hist[:, 1:], frame.unsqueeze(1)), 1)
        self.render_hist = torch.where(fresh[:, None, None, None], frame.unsqueeze(1).expand_as(hist), hist)
        lat = self.cam_params[v, 10].long().clamp(0, self.render_hist.shape[1] - 1)
        arrived = self.render_hist[torch.arange(self.n_vel, device=self.device), -1 - lat]
        rolled = torch.cat((self.depth_ring[:, 1:], arrived.unsqueeze(1)), 1)
        filled = frame.unsqueeze(1).expand_as(self.depth_ring)
        self.depth_ring = torch.where(fresh[:, None, None, None], filled, torch.where(tick[:, None, None, None], rolled, self.depth_ring))
        self.fresh[v] = False

    # ------------------------------------------------------------------ commands
    def resample_cmd(self, env_ids: torch.Tensor):
        n = len(env_ids)
        if n == 0 or not hasattr(self, "task"):
            return super().resample_cmd(env_ids)
        st, dev = self.setup, self.device
        flat = self.kind[env_ids] == FLAT_KIND
        super().resample_cmd(env_ids[flat])  # flat tiles: the flat-ground interface's commands (with reference commands)
        self.heading_mode[env_ids[flat]] = False
        self.standing[env_ids[flat]] = False
        ids = env_ids[~flat]
        m = len(ids)
        if m:
            u = lambda r: torch.empty(m, device=dev).uniform_(*r)  # noqa: E731
            vel = torch.stack((u(st.cmd_x), u(st.cmd_y), u(st.cmd_yaw)), -1)
            stairs = self.info[ids, 0] > 0.5
            fwd = stairs | (torch.rand(m, device=dev) < st.forward)
            vel = torch.where(fwd[:, None], torch.stack((u(st.forward_x), torch.zeros(m, device=dev), torch.zeros(m, device=dev)), -1), vel)
            heading = fwd | (torch.rand(m, device=dev) < st.heading)
            self.heading_mode[ids] = heading
            self.heading_target[ids] = torch.where(fwd, 0.0, u((-math.pi, math.pi)))
            self.standing[ids] = torch.rand(m, device=dev) < st.standing
            vel = torch.where(self.standing[ids, None], 0.0, vel)
            self.cmd_ref[ids] = False
            if self.mdp.upper_cmd:
                tmp = self.cmd[ids].clone()
                super().resample_cmd(ids)  # draws a fresh upper-body target (and a velocity we overwrite)
                self.cmd_ref[ids] = False
                tmp[:, 3:] = self.cmd[ids, 3:]
                self.cmd[ids] = tmp
            self.cmd[ids, :3] = vel
        self.cmd_timer[env_ids] = (torch.empty(n, device=dev).uniform_(*st.resample_s) / 0.02).long()
        self._heading_control(env_ids)

    def _cmd_due(self) -> torch.Tensor:
        return (self.is_vel & (self.cmd_timer <= 0)).nonzero(as_tuple=False).squeeze(-1)

    def _heading_control(self, ids: torch.Tensor | None = None):
        st = self.setup
        sel = self.heading_mode & self.is_vel & ~self.standing
        if ids is not None:
            mask = torch.zeros_like(sel)
            mask[ids] = True
            sel = sel & mask
        R = self.state()[1]
        err = self.heading_target - torch.atan2(R[:, 1, 0], R[:, 0, 0])
        rate = (st.heading_stiffness * torch.atan2(err.sin(), err.cos())).clamp(*st.cmd_yaw)
        self.cmd[:, 2] = torch.where(sel, rate, self.cmd[:, 2])

    # ------------------------------------------------------------------ step hooks
    def vel_reward(self, S1, cmd, a, a_prev):
        return self.task.reward(S1, cmd, a, a_prev, self.info)

    def vel_fall(self, S1):
        return self.task.fall(S1)

    def vel_timeout(self, f, clip_end, ep_step, S1):
        xy = S1[0][:, :2]
        out = ((xy < self.map_lo) | (xy > self.map_hi)).any(-1)
        clip = self.cmd_ref & (f + 1 >= clip_end)  # reference commands end with their clip
        return (ep_step + 1 >= (self.setup.episode_steps if not self.eval_mode else 10**9)) | out | clip

    def _pre_step_out(self, out: dict):
        out["info"] = self.info.clone()
        out["cam"] = self.cam_params.clone()
        out["tick"] = self.tick.clone()
        out["depth"] = encode_u8(self.depth_ring[:, -1].float())

    def _on_done(self, ids: torch.Tensor):
        st = self.setup
        vel = ids[self.is_vel[ids]]
        if not st.curriculum or self.eval_mode or not len(vel):
            return
        dist = (self.qpos[vel, :2] - self.start_xy[vel]).norm(dim=-1)
        up = dist > st.terrain.tile_size[0] / 2
        down = (dist < self.cmd[vel, :2].norm(dim=-1) * st.episode_steps * 0.02 * 0.5) & ~up
        lvl = self.level[vel] + up.long() - down.long()
        nr = self.origins.shape[0]
        lvl = torch.where(lvl >= nr, torch.randint(0, nr, lvl.shape, device=self.device), lvl.clamp(min=0))
        self.level[vel] = lvl
        self._update_info()

    def step(self, a: torch.Tensor) -> dict:
        self._in_step = True
        try:
            return super().step(a)
        finally:
            self._in_step = False

    def _post_step(self, done: torch.Tensor, ids: torch.Tensor | None):
        self.cmd_timer -= 1
        self._heading_control()
        self.cmd[:, :3] = torch.where((self.standing & self.is_vel)[:, None], 0.0, self.cmd[:, :3])
        tick = arrives(self.ep_step, self.cam_params) & self.is_vel
        if self.setup.camera.frame_drop > 0 and not self.eval_mode:
            tick = tick & (torch.rand(self.N, device=self.device) >= self.setup.camera.frame_drop)
        self.tick = tick | self.fresh
        self._update_frames(tick)

    # ------------------------------------------------------------------ stats
    def body_forces(self) -> torch.Tensor:
        """(N, nbody, 3) world-frame contact force on each body from all its contacts at the last substep
        (stairs.body_contact_vectors decodes the pyramidal forces; it scales by 1/4 for a substep average)."""
        import warp as wp

        from .stairs import body_contact_vectors

        c = self.data.contact
        with wp.ScopedDevice(self.wp_device):
            out = wp.zeros((self.N, self.mjm.nbody, 3), dtype=float)
            wp.launch(body_contact_vectors, dim=self.data.naconmax,
                      inputs=[self.data.nacon, c.worldid, c.geom, c.efc_address, self.data.efc.force,
                              self.model.geom_bodyid, c.frame, c.friction, out])  # fmt: skip
        return wp.to_torch(out) * 4

    def level_stats(self) -> dict:
        v = self.is_vel
        out = {"level_mean": self.level[v].float().mean()}
        for k in sorted(set(self.kind[v].tolist())):
            sel = v & (self.kind == k)
            out[f"level_{TILE_TYPES[k]}"] = self.level[sel].float().mean()
        return out


__all__ = ["TerrainSim", "TerrainSetup", "CAM_PARAM_DIM"]
