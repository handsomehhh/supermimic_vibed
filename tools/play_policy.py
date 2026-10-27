"""Run a trained policy live in MuJoCo (CPU) with the interactive viewer: steer the velocity interface
with the keyboard, or watch the motion tracker follow reference clips.

The physics model is the training one (supermimic/robot.py: same actuators, 4 x 5 ms substeps,
implicitfast) under evaluation settings (friction 1.0, no randomization, no noise); only the solver
runs on the CPU instead of MuJoCo-Warp.

  python tools/play_policy.py data/play/exp3_policy.pt --packs data/play/play_clips.pt data/play/lafan_loco.pt

Keys   Enter / End   next / previous episode (velocity control, then each clip)   Backspace  restart
       Space         pause                                                          G          ghost on / off
       velocity:  Up / Down  forward speed +- 0.25 m/s    PgUp / PgDn  sideways +- 0.25 m/s
                  Left / Right  turn left / right (+- 0.5 rad/s)    Home  stop (zero command)
"""

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import supermimic.networks as networks
from scene import two_robot_xml
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, OBS_DIM, REL_DIM
from supermimic.motion import CONTROL_DT, MotionLib
from supermimic.networks import Policy
from supermimic.robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list, sim_model
from supermimic.rotations import mat_to_quat, quat_to_mat

KEY_SPACE, KEY_ENTER, KEY_BACKSPACE = 32, 257, 259
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_PGUP, KEY_PGDN, KEY_HOME, KEY_END = 262, 263, 264, 265, 266, 267, 268, 269
CMD_LO, CMD_HI = np.array([-1.5, -1.32, -3.35]), np.array([2.0, 1.32, 3.35])
SUBSTEPS = 4


class LivePolicy:
    def __init__(self, ckpt_path: str, packs: list[str], device: str):
        self.dev = dev = device
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        ra = ck["args"]
        self.label = f"{os.path.basename(ckpt_path)} (it {ck['iteration']})"
        self.K, self.Hh = ra["goal_frames"], ra["hist"]
        networks.AMP_DTYPE = None  # fp32 for inference
        self.pi = Policy(OBS_DIM, 73 * self.K + REL_DIM, hist=self.Hh).to(dev)
        self.pi.load_state_dict(ck["pi"])
        self.pi.eval()
        kin = G1Kinematics(dev)
        self.motion = MotionLib.from_packs(packs, [], kin, dev)
        self.mdp = MDP(kin, self.motion, torch.tensor(default_joint_pos_list(), device=dev))
        self.q0 = np.array(default_joint_pos_list())
        self.alpha = np.array([action_scale(n) for n in G1_JOINT_NAMES])
        self.q_lo, self.q_hi = kin.q_lo.cpu().numpy(), kin.q_hi.cpu().numpy()
        self.xi = torch.zeros(1, 33, device=dev)
        self.xi[:, 0] = 1.0
        self.m = sim_model(CONTROL_DT / SUBSTEPS)
        self.d = mujoco.MjData(self.m)
        self.stand_height = self._stand_height()

    def _stand_height(self) -> float:
        d = mujoco.MjData(self.m)
        d.qpos[2], d.qpos[3], d.qpos[7:] = 1.0, 1.0, self.q0
        mujoco.mj_kinematics(self.m, d)
        spheres = (self.m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) & (self.m.geom_bodyid > 0)
        return float(1.0 - (d.geom_xpos[spheres, 2] - self.m.geom_size[spheres, 0]).min() + 0.002)

    # ---------------------------------------------------------------- state
    def state(self):
        t = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=self.dev).unsqueeze(0)  # noqa: E731
        R = quat_to_mat(t(self.d.qpos[3:7]))
        w = (R @ t(self.d.qvel[3:6]).unsqueeze(-1)).squeeze(-1)  # free-joint angular velocity is body-local
        return t(self.d.qpos[0:3]), R, t(self.d.qvel[0:3]), w, t(self.d.qpos[7:36]), t(self.d.qvel[6:35])

    def reset(self, frame: int | None):
        """frame None: standing in the default pose; else the reference state of that motion frame."""
        mujoco.mj_resetData(self.m, self.d)
        if frame is None:
            self.d.qpos[0:3], self.d.qpos[3:7], self.d.qpos[7:36] = (0, 0, self.stand_height), (1, 0, 0, 0), self.q0
        else:
            f = torch.tensor([frame], device=self.dev)
            p, R, v, w, q, qd = (x[0].cpu().numpy() for x in self.motion.ref_state(f))
            q = np.clip(q, self.q_lo, self.q_hi)
            self.d.qpos[0:3], self.d.qpos[3:7], self.d.qpos[7:36] = p, mat_to_quat(torch.tensor(R)).numpy(), q
            self.d.qvel[0:3], self.d.qvel[3:6], self.d.qvel[6:35] = v, R.T @ w, qd
        self.d.ctrl[:] = self.d.qpos[7:36]
        mujoco.mj_forward(self.m, self.d)
        self.a_prev = torch.zeros(1, 29, device=self.dev)
        self.hist = None

    @torch.no_grad()
    def act(self, f: int | None, cmd: np.ndarray | None) -> None:
        """One 50 Hz control step: motion tracking toward frame f, or the velocity command cmd."""
        S = self.state()
        o = self.mdp.obs(S, self.a_prev, self.xi)
        self.hist = o.unsqueeze(1).repeat(1, self.Hh, 1) if self.hist is None else torch.cat((self.hist[:, 1:], o.unsqueeze(1)), 1)
        if cmd is None:
            g = self.mdp.motion_goal(S[1], torch.tensor([f], device=self.dev), self.K, noise=False)
            a = self.pi(self.hist, g)
        else:
            a = self.pi.act_vel(self.hist, self.mdp.norm_cmd(torch.tensor(cmd, dtype=torch.float32, device=self.dev).unsqueeze(0)))
        self.a_prev = a
        self.d.ctrl[:] = np.clip(self.q0 + self.alpha * a[0].cpu().numpy(), self.q_lo, self.q_hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.m, self.d)


class App:
    def __init__(self, lp: LivePolicy, clips: list[int]):
        self.lp, self.clips = lp, clips
        self.vm = mujoco.MjModel.from_xml_string(two_robot_xml())
        self.vm.vis.headlight.ambient[:] = 0.45
        self.vm.vis.headlight.diffuse[:] = 0.55
        self.vd = mujoco.MjData(self.vm)
        self.ep, self.paused, self.ghost = 0, False, True
        self.cmd = np.zeros(3)
        self.start_episode()

    @property
    def is_vel(self) -> bool:
        return self.ep == 0

    def start_episode(self):
        m = self.lp.motion
        if self.is_vel:
            self.lp.reset(None)
            self.f = None
        else:
            c = self.clips[self.ep - 1]
            self.f = int(m.clip_start[c])
            self.f_end = int(m.clip_end[c])
            self.lp.reset(self.f)
        self.terminated, self.t = False, 0.0

    def on_key(self, key: int):
        n = len(self.clips) + 1
        if key == KEY_SPACE:
            self.paused = not self.paused
        elif key in (KEY_ENTER, KEY_END):
            self.ep = (self.ep + (1 if key == KEY_ENTER else -1)) % n
            self.start_episode()
        elif key == KEY_BACKSPACE:
            self.start_episode()
        elif key == ord("G"):
            self.ghost = not self.ghost
        elif self.is_vel and key in (KEY_UP, KEY_DOWN, KEY_PGUP, KEY_PGDN, KEY_LEFT, KEY_RIGHT, KEY_HOME):
            if key == KEY_HOME:
                self.cmd[:] = 0
            else:
                i, step = {KEY_UP: (0, 0.25), KEY_DOWN: (0, -0.25), KEY_PGUP: (1, 0.25), KEY_PGDN: (1, -0.25),
                           KEY_LEFT: (2, 0.5), KEY_RIGHT: (2, -0.5)}[key]  # fmt: skip
                self.cmd[i] = np.clip(self.cmd[i] + step, CMD_LO[i], CMD_HI[i])

    def control_step(self):
        lp = self.lp
        if self.is_vel:
            lp.act(None, self.cmd)
            if lp.mdp.fall(lp.state()).item():
                self.terminated = True
        else:
            lp.act(self.f, None)
            self.f = min(self.f + 1, self.f_end)
            if lp.mdp.termination(lp.state(), torch.tensor([self.f], device=lp.dev)).item():
                self.terminated = True
            if self.f >= self.f_end:
                self.start_episode()  # loop the clip
        self.t += CONTROL_DT

    def overlay(self):
        lp = self.lp
        rows = [("policy", lp.label)]
        if self.is_vel:
            S = lp.state()
            meas = lp.mdp.heading_vel(S)[0].cpu().numpy()
            rows += [("mode", "velocity commands (keyboard)"),
                     ("command", f"vx {self.cmd[0]:+.2f}  vy {self.cmd[1]:+.2f}  yaw {self.cmd[2]:+.2f}"),
                     ("measured", f"vx {meas[0]:+.2f}  vy {meas[1]:+.2f}  yaw {meas[2]:+.2f}"),
                     ("keys", "Up/Down fwd  PgUp/PgDn side  Left/Right turn  Home stop")]  # fmt: skip
        else:
            c = self.clips[self.ep - 1]
            m = lp.motion
            rows += [("mode", f"motion tracking  clip {self.ep}/{len(self.clips)}"), ("clip", m.names[c]),
                     ("clip time", f"{(self.f - int(m.clip_start[c])) * CONTROL_DT:5.1f} s / {(int(m.clip_len[c]) - 1) * CONTROL_DT:.1f} s")]  # fmt: skip
        if self.terminated:
            rows.append(("state", "FELL / TERMINATED  (Backspace restarts)"))
        rows.append(("episode", "Enter / End next / previous   Space pause   G ghost" + ("   [paused]" if self.paused else "")))
        return rows

    def sync_view(self, viewer):
        q = self.vd.qpos
        q[0:36] = self.lp.d.qpos
        if self.is_vel or not self.ghost:
            q[36:39] = (0, 0, -10)
        else:
            f = torch.tensor([self.f], device=self.lp.dev)
            m = self.lp.motion
            q[36:39] = m.root_p[f][0].cpu().numpy()
            q[36:38] = q[0:2]  # reference placed at the robot's xy, as the tracking reward compares it
            q[39:43] = mat_to_quat(m.root_R[f])[0].cpu().numpy()
            q[43:72] = m.q[f][0].cpu().numpy()
        mujoco.mj_kinematics(self.vm, self.vd)
        viewer.cam.lookat[:] = [q[0], q[1], 0.75 if not self.is_vel else 0.5]
        scn = viewer.user_scn
        scn.ngeom = 0
        if self.is_vel:
            w, x, y, z = q[3:7]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            H = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            meas = self.lp.mdp.heading_vel(self.lp.state())[0].cpu().numpy()
            base = np.array([q[0], q[1], 0.02])
            for v, rgba in ((self.cmd, (0.15, 0.75, 0.35, 0.9)), (meas, (1.0, 0.55, 0.1, 0.9))):
                vec = np.r_[H @ v[:2], 0.0]
                if np.linalg.norm(vec) < 1e-3:
                    continue
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.04, base, base + vec)
                scn.ngeom += 1
        rows = self.overlay()
        viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                          "\n".join(k for k, _ in rows), "\n".join(v for _, v in rows)))  # fmt: skip

    def run(self):
        with mujoco.viewer.launch_passive(self.vm, self.vd, key_callback=self.on_key) as viewer:
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.6, 135.0, -15.0
            viewer.opt.geomgroup[3] = 0
            next_t = time.perf_counter()
            while viewer.is_running():
                with viewer.lock():
                    if not self.paused:
                        self.control_step()
                    self.sync_view(viewer)
                viewer.sync()
                next_t += CONTROL_DT
                time.sleep(max(0.0, next_t - time.perf_counter()))
                if time.perf_counter() - next_t > 0.5:  # fell behind (e.g. window dragged): resync the clock
                    next_t = time.perf_counter()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt")
    ap.add_argument("--packs", nargs="+", default=[os.path.join(REPO, "data", "play", "play_clips.pt"), os.path.join(REPO, "data", "play", "lafan_loco.pt")])
    ap.add_argument("--clips", nargs="*", default=[], help="only clips whose name contains one of these (default: all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--headless", type=int, default=0, help="run N control steps without a window and print a check")
    a = ap.parse_args()
    lp = LivePolicy(a.ckpt, a.packs, a.device)
    names = lp.motion.names
    clips = [i for i, n in enumerate(names) if not a.clips or any(s in n for s in a.clips)]
    print(f"{lp.label}: {len(clips)} clips, motion library {lp.motion.num_frames * CONTROL_DT / 60:.1f} min")
    app = App(lp, clips)
    if a.headless:
        app.cmd[:] = (1.0, 0.0, 0.0)
        for _ in range(a.headless):
            app.control_step()
        meas = lp.mdp.heading_vel(lp.state())[0].cpu().numpy()
        print(f"velocity check after {a.headless * CONTROL_DT:.0f} s at vx=1.0: measured {np.round(meas, 2)}, fell {app.terminated}")
        fails = 0
        for k in range(1, min(len(clips), 8) + 1):
            app.ep = k
            app.start_episode()
            steps = min(int(lp.motion.clip_len[clips[k - 1]]) - 2, 500)
            for _ in range(steps):
                app.control_step()
            fails += app.terminated
            print(f"  tracking {names[clips[k - 1]][:50]:50s} {steps * CONTROL_DT:4.1f} s  terminated {app.terminated}")
        print(f"tracking check: {fails} of {min(len(clips), 8)} clips terminated")
        return
    print(__doc__.split("Keys", 1)[1].join(("Keys", "")))
    app.run()


if __name__ == "__main__":
    main()
