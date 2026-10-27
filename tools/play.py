"""Replay recorded rollouts (tools/record.py) in the interactive MuJoCo viewer.

  python tools/play.py rec/exp3.npz [rec/exp2.npz ...]

Episodes: every recorded motion-tracking clip (solid robot + blue reference ghost) and every velocity-
command sequence (green arrow: commanded velocity, orange: measured; yaw rates in the overlay).

Keys   Space        pause / play
       Right / Left paused: one frame forward / back; playing: jump 2 s
       PgDn / PgUp  next / previous episode        End / Home  next / previous recording
       Up / Down    speed x2 / /2                  Backspace   restart the episode
       G            ghost: anchored to the robot / at the reference's own place / hidden
       Enter        camera follows the robot on / off
       (the viewer's own keys keep working: Tab hides the panels, double-click selects, ...)
"""

import argparse
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scene import two_robot_xml

KEY_SPACE, KEY_ENTER, KEY_BACKSPACE = 32, 257, 259
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_PGUP, KEY_PGDN, KEY_HOME, KEY_END = 262, 263, 264, 265, 266, 267, 268, 269
GHOST_MODES = ("anchored", "global", "hidden")


class Recording:
    def __init__(self, path: str):
        d = dict(np.load(path, allow_pickle=False))
        self.label = str(d["label"])
        self.dt = float(d.get("dt", 0.02))
        self.episodes = []  # (kind, index, name, length)
        for c, name in enumerate(d["trk_names"]):
            self.episodes.append(("trk", c, str(name), int(d["trk_len"][c])))
        for e, name in enumerate(d.get("vel_names", [])):
            self.episodes.append(("vel", e, str(name), d["vel_root_p"].shape[0]))
        self.d = d


class Player:
    def __init__(self, recs: list[Recording]):
        self.recs, self.r, self.e = recs, 0, 0
        self.model = mujoco.MjModel.from_xml_string(two_robot_xml())
        self.model.vis.headlight.ambient[:] = 0.45
        self.model.vis.headlight.diffuse[:] = 0.55
        self.data = mujoco.MjData(self.model)
        self.t, self.paused, self.speed, self.ghost, self.follow, self.step_req = 0.0, False, 1.0, 0, True, 0

    @property
    def rec(self) -> Recording:
        return self.recs[self.r]

    @property
    def ep(self):
        return self.rec.episodes[self.e]

    def on_key(self, key: int):
        n_ep = len(self.rec.episodes)
        if key == KEY_SPACE:
            self.paused = not self.paused
        elif key in (KEY_RIGHT, KEY_LEFT):
            sign = 1 if key == KEY_RIGHT else -1
            if self.paused:
                self.step_req = sign
            else:
                self.t = max(self.t + 2.0 * sign, 0.0)
        elif key in (KEY_PGDN, KEY_PGUP):
            self.e, self.t = (self.e + (1 if key == KEY_PGDN else -1)) % n_ep, 0.0
        elif key in (KEY_END, KEY_HOME):
            self.r, self.e, self.t = (self.r + (1 if key == KEY_END else -1)) % len(self.recs), 0, 0.0
        elif key == KEY_UP:
            self.speed = min(self.speed * 2, 8.0)
        elif key == KEY_DOWN:
            self.speed = max(self.speed / 2, 0.125)
        elif key == KEY_BACKSPACE:
            self.t = 0.0
        elif key == KEY_ENTER:
            self.follow = not self.follow
        elif key == ord("G"):
            self.ghost = (self.ghost + 1) % len(GHOST_MODES)

    def pose(self, i: int):
        """Write frame i of the current episode into qpos; returns overlay lines and the arrows to draw."""
        kind, c, name, n = self.ep
        d, q = self.rec.d, self.data.qpos
        p = kind + "_"
        q[0:3], q[3:7], q[7:36] = d[p + "root_p"][i, c], d[p + "root_quat"][i, c], d[p + "q"][i, c]
        alive = bool(d[p + "alive"][i, c])
        lines, arrows = [], []
        if kind == "trk":
            q[36:39], q[39:43], q[43:72] = d["trk_ref_root_p"][i, c], d["trk_ref_root_quat"][i, c], d["trk_ref_q"][i, c]
            if GHOST_MODES[self.ghost] == "anchored":  # reference placed at the robot's xy, as the tracking reward compares it
                q[36:38] = q[0:2]
            elif GHOST_MODES[self.ghost] == "hidden":
                q[36:39] = (0, 0, -10)
            ret, failed = float(d["trk_return"][c]), bool(d["trk_failed"][c])
            lines += [("motion tracking", f"{name}  ({d['trk_why'][c]})"),
                      ("clip time", f"{float(d['trk_t'][i, c]):5.1f} s / {n * self.rec.dt:.1f} s"),
                      ("clip result", f"return {ret:.3f}" + ("   TERMINATED" if failed else "")),
                      ("ghost", GHOST_MODES[self.ghost])]  # fmt: skip
        else:
            q[36:39] = (0, 0, -10)
            cmd, meas = d["vel_cmd"][i, c], d["vel_meas"][i, c]
            lines += [("velocity commands", name), ("segment", str(d["vel_label"][i, c])),
                      ("command", f"vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  yaw {cmd[2]:+.2f}"),
                      ("measured", f"vx {meas[0]:+.2f}  vy {meas[1]:+.2f}  yaw {meas[2]:+.2f}"),
                      ("time", f"{i * self.rec.dt:5.1f} s")]  # fmt: skip
            w, x, y, z = q[3:7]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            H = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            base = np.array([q[0], q[1], 0.02])
            for v, rgba in ((cmd, (0.15, 0.75, 0.35, 0.9)), (meas, (1.0, 0.55, 0.1, 0.9))):
                arrows.append((base, np.r_[H @ v[:2], 0.0], rgba))
        if not alive:
            lines.append(("state", "FELL / TERMINATED"))
        lines.append(("playback", f"{'paused' if self.paused else f'x{self.speed:g}'}   recording {self.r + 1}/{len(self.recs)}: {self.rec.label}"))
        lines.append(("episode", f"{self.e + 1}/{len(self.rec.episodes)}   (PgDn/PgUp episode, End/Home recording, G ghost)"))
        return lines, arrows

    def draw_arrows(self, viewer, arrows):
        scn = viewer.user_scn
        scn.ngeom = 0
        for base, vec, rgba in arrows:
            if np.linalg.norm(vec) < 1e-3 or scn.ngeom >= scn.maxgeom:
                continue
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.025, base, base + vec)
            scn.ngeom += 1

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.on_key) as viewer:
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.6, 135.0, -12.0
            last = time.time()
            while viewer.is_running():
                now = time.time()
                n = self.ep[3]
                if self.step_req:
                    self.t = min(max(self.t + self.step_req * self.rec.dt, 0.0), (n - 1) * self.rec.dt)
                    self.step_req = 0
                elif not self.paused:
                    self.t += (now - last) * self.speed
                    if self.t >= n * self.rec.dt:
                        self.t = 0.0  # loop the episode
                last = now
                i = min(int(self.t / self.rec.dt), n - 1)
                lines, arrows = self.pose(i)
                mujoco.mj_kinematics(self.model, self.data)
                with viewer.lock():
                    viewer.opt.geomgroup[2] = GHOST_MODES[self.ghost] != "hidden"
                    if self.follow:
                        viewer.cam.lookat[:] = [self.data.qpos[0], self.data.qpos[1], 0.75]
                    self.draw_arrows(viewer, arrows)
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      "\n".join(k for k, _ in lines), "\n".join(v for _, v in lines)))  # fmt: skip
                viewer.sync()
                time.sleep(1 / 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+")
    a = ap.parse_args()
    recs = [Recording(p) for p in a.recordings]
    for r in recs:
        print(f"{r.label}: " + ", ".join(f"{e[2]}" for e in r.episodes))
    print(__doc__.split("Keys", 1)[1].join(("Keys", "")))
    Player(recs).run()


if __name__ == "__main__":
    main()
