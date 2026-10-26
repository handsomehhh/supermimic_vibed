"""Motion-tracking / velocity-command simulator on MuJoCo-Warp (50 Hz control, 4 x 5 ms substeps).

The environment is a thin batched wrapper: it applies the joint setpoint, steps MuJoCo-Warp (one CUDA
graph per control step), labels the transition with the analytic reward/termination of
:mod:`supermimic.mdp`, and resets episodes to reference-motion states. Environments
``[N - n_vel, N)`` run the velocity interface (paper App. G.1); the rest track motions, reset by
return-based bin sampling (App. D.1).
"""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from .mdp import MAX_STEP_REWARD, MAX_STEP_REWARD_VEL, MDP, flatten_state
from .motion import CONTROL_DT, MotionLib
from .robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list, sim_model
from .rotations import mat_to_quat, quat_to_mat, so3_exp

# velocity command ranges (App. G.1)
CMD_LO = (-1.5, -1.32, -3.35)
CMD_HI = (2.0, 1.32, 3.35)
CMD_RESAMPLE_STEPS = 250
# a world whose velocities exceed these bounds (or turn non-finite) has blown up numerically
MAX_ROOT_VEL, MAX_JOINT_VEL = 50.0, 100.0


def use_warp_stream(device: str):
    """Run torch on Warp's stream so simulator kernels and torch ops are ordered without syncs."""
    wp.init()
    torch.cuda.set_device(device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(wp.get_device(device)).cuda_stream, device=device))


@wp.kernel
def _repeat(src: wp.array(dtype=wp.vec3), n: int, dst: wp.array(dtype=wp.vec3)):
    t = wp.tid()
    dst[t] = src[t % n]


def _expand(model, name: str, nworld: int):
    """Give a (1, n) vec3 model field one row per world."""
    a = getattr(model, name)
    dst = wp.empty((nworld, a.shape[1]), dtype=a.dtype, device=a.device)
    wp.launch(_repeat, dim=nworld * a.shape[1], inputs=[a.flatten(), a.shape[1]], outputs=[dst.flatten()], device=a.device)
    setattr(model, name, dst)
    return wp.to_torch(dst)


class TrackingSim:
    """Batched G1 simulator. Call :func:`use_warp_stream` before constructing it."""

    def __init__(self, num_envs: int, motion: MotionLib, mdp: MDP, device: str, gamma: float = 0.99, eval_mode: bool = False,
                 max_episode_steps: int = 500, reset_noise: bool = True, n_vel: int = 0, substeps: int = 4, reset_temp: float = 25.0):  # fmt: skip
        self.device = dev = device
        self.N, self.n_vel = num_envs, n_vel
        self.reset_temp = reset_temp  # softmax temperature of return-based reset sampling (paper: 25)
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.max_episode_steps = max_episode_steps if not eval_mode else 10**9
        self.reset_noise = reset_noise and not eval_mode
        self.mjm = mjm = sim_model(CONTROL_DT / substeps)
        assert [mjm.joint(i).name for i in range(1, mjm.njnt)] == G1_JOINT_NAMES
        self.wp_device = wp.get_device(dev)
        with wp.ScopedDevice(self.wp_device):
            self.model = mjw.put_model(mjm)
            self.data = mjw.make_data(mjm, nworld=num_envs, nconmax=128, njmax=600)
            friction = _expand(self.model, "geom_friction", num_envs)
            ipos = _expand(self.model, "body_ipos", num_envs)
            self._blown = wp.zeros(num_envs, dtype=bool)
        self.qpos, self.qvel, self.ctrl = (wp.to_torch(x) for x in (self.data.qpos, self.data.qvel, self.data.ctrl))
        self.blown = wp.to_torch(self._blown)
        self.default_q = torch.tensor(default_joint_pos_list(), device=dev)
        self.alpha = torch.tensor([action_scale(n) for n in G1_JOINT_NAMES], device=dev)
        self.is_vel = torch.arange(num_envs, device=dev) >= num_envs - n_vel

        # ---- sim context xi = [friction, joint offsets(29), torso com offset(3)] ----
        self.xi = torch.zeros(self.N, 33, device=dev)
        robot_geoms = torch.as_tensor(mjm.geom_bodyid > 0, device=dev)
        # MuJoCo takes the max friction of a pair: the floor holds the lowest value of the range
        friction[:, ~robot_geoms, 0] = 1.0 if eval_mode else 0.3
        if eval_mode:
            self.xi[:, 0] = 1.0
        else:
            self.xi[:, 0] = torch.empty(self.N, device=dev).uniform_(0.3, 1.2)
            self.xi[:, 30:33] = torch.stack(
                (torch.empty(self.N, device=dev).uniform_(-0.025, 0.025), torch.empty(self.N, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(self.N, device=dev).uniform_(-0.05, 0.05)), -1)  # fmt: skip
            torso = mjm.body("torso_link").id
            ipos[:, torso] += self.xi[:, 30:33]
        friction[:, robot_geoms, 0] = self.xi[:, :1]
        with wp.ScopedDevice(self.wp_device), wp.ScopedCapture() as cap:
            for _ in range(substeps):
                mjw.step(self.model, self.data)
        self.step_graph = cap.graph

        # ---- episode bookkeeping ----
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, 29, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.zeros(motion.num_bins, device=dev)
        self.bin_w_avg = torch.where(motion.bin_motion, 1.0, 0.0) / motion.bin_motion.sum().clamp(min=1)
        self.cmd = torch.zeros(self.N, 3, device=dev)
        self.cmd_ref = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.cmd_lo, self.cmd_hi = torch.tensor(CMD_LO, device=dev), torch.tensor(CMD_HI, device=dev)
        self.resample_cmd_enabled = not eval_mode
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.stand_height = self._stand_height()
        self.reset(torch.arange(self.N, device=dev))

    def _stand_height(self) -> float:
        """Pelvis height at which the default pose touches the floor."""
        d = mujoco.MjData(self.mjm)
        d.qpos[2], d.qpos[3] = 1.0, 1.0
        d.qpos[7:] = default_joint_pos_list()
        mujoco.mj_kinematics(self.mjm, d)
        spheres = (self.mjm.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) & (self.mjm.geom_bodyid > 0)
        low = (d.geom_xpos[spheres, 2] - self.mjm.geom_size[spheres, 0]).min()
        return float(1.0 - low + 0.002)

    # ------------------------------------------------------------------ state
    def state(self):
        p, quat, q = self.qpos[:, :3], self.qpos[:, 3:7], self.qpos[:, 7:]
        R = quat_to_mat(quat)
        w = (R @ self.qvel[:, 3:6].unsqueeze(-1)).squeeze(-1)  # free-joint angular velocity is body-local
        return p.clone(), R, self.qvel[:, :3].clone(), w, q.clone(), self.qvel[:, 6:].clone()

    def write_state(self, env_ids, p, R, v, w, q, qd):
        self.qpos[env_ids] = torch.cat((p, mat_to_quat(R), q), -1)
        self.qvel[env_ids] = torch.cat((v, (R.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1), qd), -1)
        self.ctrl[env_ids] = q

    # ------------------------------------------------------------------ reset
    def sample_frames(self, n: int, vel: bool) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        if vel:
            probs = m.bin_loco.float()
        else:
            soft = torch.softmax(torch.where(m.bin_motion, -self.reset_temp * self.bin_G / self.G_max, -torch.inf), 0)
            self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
            probs = 0.5 * m.bin_motion.float() / m.bin_motion.sum() + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])  # keep >= 1 step before the clip end
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return f, bins

    def resample_cmd(self, env_ids: torch.Tensor):
        n = len(env_ids)
        if n == 0:
            return
        dev = self.device
        self.cmd_ref[env_ids] = torch.rand(n, device=dev) < 0.5
        u = self.cmd_lo + (self.cmd_hi - self.cmd_lo) * torch.rand(n, 3, device=dev)
        u = torch.where((torch.rand(n, 1, device=dev) < 0.1), 0.0, u)
        self.cmd[env_ids] = torch.where(self.cmd_ref[env_ids, None], self.motion.ref_cmd[self.f[env_ids]], u)

    def reset(self, env_ids: torch.Tensor, frames: torch.Tensor | None = None):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            vel = self.is_vel[env_ids]
            frames = torch.empty(n, dtype=torch.long, device=dev)
            bins = torch.empty(n, dtype=torch.long, device=dev)
            for flag in (False, True):
                idx = (vel == flag).nonzero(as_tuple=False).squeeze(-1)
                if len(idx):
                    frames[idx], bins[idx] = self.sample_frames(len(idx), flag)
        else:
            bins = self.motion.frame_bin(frames)
        p, R, v, w, q, qd = (x.clone() for x in self.motion.ref_state(frames))
        if self.reset_noise:
            p = p + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.05, 0.05), torch.empty(n, device=dev).uniform_(-0.05, 0.05),
                 torch.empty(n, device=dev).uniform_(-0.01, 0.01)), -1)  # fmt: skip
            rv = torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.1, 0.1), torch.empty(n, device=dev).uniform_(-0.1, 0.1),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            R = so3_exp(rv) @ R
            v = v + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.5, 0.5), torch.empty(n, device=dev).uniform_(-0.5, 0.5),
                 torch.empty(n, device=dev).uniform_(-0.2, 0.2)), -1)  # fmt: skip
            w = w + torch.stack(
                (torch.empty(n, device=dev).uniform_(-0.52, 0.52), torch.empty(n, device=dev).uniform_(-0.52, 0.52),
                 torch.empty(n, device=dev).uniform_(-0.78, 0.78)), -1)  # fmt: skip
            q = q + torch.empty_like(q).uniform_(-0.1, 0.1)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        self.write_state(env_ids, p, R, v, w, q, qd)
        self._reset_bookkeeping(env_ids, frames, bins)

    def reset_standing(self, env_ids: torch.Tensor, frames: torch.Tensor):
        """Default pose, upright and at rest (velocity step test, App. G.2)."""
        n = len(env_ids)
        dev = self.device
        p = torch.tensor([0.0, 0.0, self.stand_height], device=dev).expand(n, 3)
        R = torch.eye(3, device=dev).expand(n, 3, 3)
        zero = torch.zeros(n, 3, device=dev)
        q = self.default_q.expand(n, 29)
        self.write_state(env_ids, p, R, zero, zero, q, torch.zeros(n, 29, device=dev))
        self._reset_bookkeeping(env_ids, frames, self.motion.frame_bin(frames))

    def _reset_bookkeeping(self, env_ids, frames, bins):
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        if not self.eval_mode:
            self.xi[env_ids, 1:30] = torch.empty(len(env_ids), 29, device=self.device).uniform_(-0.01, 0.01)
        if self.resample_cmd_enabled:
            self.resample_cmd(env_ids[self.is_vel[env_ids]])

    # ------------------------------------------------------------------ step
    def step(self, a: torch.Tensor) -> dict:
        S = self.state()
        f = self.f.clone()
        xi = self.xi.clone()
        a_prev = self.a_prev.clone()
        ep_step = self.ep_step.clone()
        cmd, cmd_ref = self.cmd.clone(), self.cmd_ref.clone()
        # setpoints beyond the joint range only push the joint into its limit (and can blow up the solver)
        self.ctrl[:] = torch.clamp(self.default_q + xi[:, 1:30] + self.alpha * a, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        qpos0, qvel0 = self.qpos.clone(), self.qvel.clone()
        wp.capture_launch(self.step_graph)
        bad = ~(torch.isfinite(self.qpos).all(-1) & torch.isfinite(self.qvel).all(-1))
        bad |= (self.qvel[:, :6].abs().amax(-1) > MAX_ROOT_VEL) | (self.qvel[:, 6:].abs().amax(-1) > MAX_JOINT_VEL)
        # roll blown-up worlds back to their pre-step state; the transition is marked invalid and the episode ends
        self.qpos.copy_(torch.where(bad.unsqueeze(-1), qpos0, self.qpos))
        self.qvel.copy_(torch.where(bad.unsqueeze(-1), qvel0, self.qvel))
        S1 = self.state()
        clip_end = self.motion.clip_end[self.motion.clip_id[f]]
        f1 = torch.minimum(f + 1, clip_end)
        vel = self.is_vel
        r = torch.where(vel, self.mdp.vel_reward(S1, cmd, a, a_prev), self.mdp.reward(S1, f1, a, a_prev))
        r = torch.where(bad, 0.0, r)
        term = torch.where(vel, self.mdp.fall(S1), self.mdp.termination(S1, f1)) | bad
        timeout = (f + 1 >= clip_end) | (ep_step + 1 >= self.max_episode_steps)
        if self.eval_mode:
            timeout = timeout & ~vel  # the step test ends on its own clock
        done = term | timeout
        out = {
            "s": flatten_state(S), "a": a.clone(), "a_prev": a_prev, "f": f, "s1": flatten_state(S1), "r": r,
            "term": term, "done": done, "ep_step": ep_step, "xi": xi, "cmd": cmd, "cmd_ref": cmd_ref, "valid": ~bad,
            "n_blown": bad.sum(),
        }  # fmt: skip
        # bookkeeping
        self.ep_disc_ret += (self.gamma ** ep_step.float()) * r
        self.ep_ret += r
        self.f = f1
        self.ep_step = ep_step + 1
        self.a_prev = a.clone()
        if self.resample_cmd_enabled:
            self.cmd[cmd_ref] = self.motion.ref_cmd[f1[cmd_ref]]
            due = (vel & (self.ep_step % CMD_RESAMPLE_STEPS == 0)).nonzero(as_tuple=False).squeeze(-1)
            self.resample_cmd(due)
        if self.eval_mode:
            self.failed_eval |= term & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            T = self.ep_step[ids].float()
            mo = ids[~vel[ids] & ~bad[ids]]
            if len(mo) > 0:
                # bin return estimate; truncated episodes bootstrap with their mean reward
                Tm = self.ep_step[mo].float()
                boot = torch.where(term[mo], 0.0, (self.gamma**Tm) * (self.ep_ret[mo] / Tm) / (1 - self.gamma))
                b = self.start_bin[mo]
                self.bin_G.index_put_((b,), 0.8 * self.bin_G[b] + 0.2 * (self.ep_disc_ret[mo] + boot))
            out["done_ids"] = ids
            out["done_vel"] = vel[ids]
            out["ep_len"] = T
            out["ep_ret_norm"] = self.ep_ret[ids] / (T * torch.where(vel[ids], MAX_STEP_REWARD_VEL, MAX_STEP_REWARD))
            if bool(bad.any()):  # clear the solver state (warm start, contacts) of blown-up worlds
                self.blown.copy_(bad)
                with wp.ScopedDevice(self.wp_device):
                    mjw.reset_data(self.model, self.data, reset=self._blown)
            self.reset(ids)
        return out
