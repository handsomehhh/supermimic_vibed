"""Play loco teachers or map students in the training MuJoCo-Warp environment.

The CPU model only displays GPU world 0. Evaluation settings match eval_loco:
no curriculum, randomization or proprioceptive noise; all seven terrain types.
Start facing +x with vx=0.6 m/s (stairs_down is a pit, so walking out climbs).

  python tools/play_loco.py logs/loco/student_cnn_stairs_scratchT/ckpt_005064.pt --terrain stairs_down --level 8

Keys   Up / Down       forward speed +- 0.1 m/s    PgUp / PgDn  sideways +- 0.1 m/s
       Left / Right    yaw rate +- 0.5 rad/s       Home         zero command
       Space           pause                      Backspace    reset selected tile
       1..7 or [ / ]   select / cycle tile type    - / =        difficulty 0..9
       M               student elevation map on / off
"""
from __future__ import annotations

import argparse
from copy import copy, deepcopy
from pathlib import Path
from queue import SimpleQueue
import sys
import time

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supermimic.loco_env import LocoEnv, setup_device
from supermimic.student import is_student, load_student, student_inputs
from supermimic.terrain import TILE_TYPES, write_hfield
from supermimic.train_ppo import LocoRunner, cfg_from_dict

KEY_SPACE, KEY_BACKSPACE = 32, 259
KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP, KEY_PGUP, KEY_PGDN, KEY_HOME = 262, 263, 264, 265, 266, 267, 268
COMMAND_KEYS = {KEY_UP: (0, .1), KEY_DOWN: (0, -.1), KEY_PGUP: (1, .1), KEY_PGDN: (1, -.1),
                KEY_LEFT: (2, .5), KEY_RIGHT: (2, -.5)}
CMD_LO, CMD_HI = (-1.5, -1.32, -3.35), (2., 1.32, 3.35)


class Player:
    def __init__(self, checkpoint, terrain, level, map_noise, device):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        torch.manual_seed(0)
        cfg = cfg_from_dict(deepcopy(ck["env_cfg"]))
        cfg.curriculum = cfg.randomization = cfg.observation_noise = False
        self.student = is_student(ck)
        if self.student:
            if "elevation_map" not in student_inputs(ck):
                raise ValueError("Expected an elevation-map student checkpoint")
            cfg.elevation.enabled, cfg.elevation.noise = True, map_noise
        cfg.episode_seconds = 20
        # Match eval_loco, including checkpoints trained on only one terrain type.
        cfg.terrain.proportions = (.10, .20, .15, .15, .15, .15, .10)
        cfg.terrain.num_cols = max(20, cfg.terrain.num_cols)
        self.max_level = min(9, cfg.terrain.num_rows - 1)
        if not 0 <= level <= self.max_level:
            raise ValueError(f"Checkpoint terrain supports levels 0..{self.max_level}")
        self.env = env = LocoEnv(1, cfg, device, eval_mode=True)
        self.kind, self.level = TILE_TYPES.index(terrain), level
        self.columns = [int((env.tile_types[0] == k).nonzero()[0, 0]) for k in range(len(TILE_TYPES))]
        self.cmd = np.array([.6, 0., 0.])
        self.paused, self.show_map = False, False
        self.keys = SimpleQueue()
        self.reset()
        if self.student:
            self.policy = load_student(ck, device)
        else:
            runner = LocoRunner(env, ck["runner_cfg"], device=device)
            runner.alg.load(ck, {"actor": True, "critic": True}, True)
            self.policy = runner.get_inference_policy(device)
        self.label = f"{Path(checkpoint).name} ({'student' if self.student else 'teacher'}, it {ck.get('iter', '?')})"
        # An independent CPU display model; never advance physics with mj_step.
        self.vm = copy(env.mjm)
        write_hfield(self.vm, env.terrain)
        self.vm.vis.headlight.ambient[:] = .45
        self.vm.vis.headlight.diffuse[:] = .55
        self.vd = mujoco.MjData(self.vm)
        self.sync_model()

    def reset(self):
        env = self.env
        env.levels.fill_(self.level)
        env.columns.fill_(self.columns[self.kind])
        env.kinds.fill_(self.kind)
        env.type_masks = {name: env.kinds == k for k, name in enumerate(TILE_TYPES)}
        env.reset(env.ids)
        self.start = env.qpos[0, :2].clone()
        self.distance, self.steps = 0., 0
        self.fell = self.last_fall = False
        self.map_stamp = -1
        self.map_points = np.empty((0, 3))
        self.set_command()
        self.cache_map()

    def set_command(self):
        env = self.env
        env.cmd[:] = torch.as_tensor(self.cmd, device=env.device, dtype=env.cmd.dtype)
        # get_observations does not advance history twice in one control tick.
        # Update only the current frame, leaving past commands intact.
        if env.elevation is not None or env.depth is not None:
            env.policy_history[:, -1, 6:9] = env.cmd
        self.obs = env.get_observations()

    def cache_map(self):
        elevation = self.env.elevation
        if not self.student or elevation is None:
            return
        stamp = int(elevation.refresh_step[0])
        if stamp == self.map_stamp:
            return
        self.map_stamp = stamp
        # Cache before the pose ring overwrites the acquisition frame. Draw the
        # observed heights at nominal cell locations, retaining sensor errors.
        index = int(elevation.acquisition_step[0]) % elevation.ring_size
        pose = elevation.poses[0, index].cpu().numpy()
        values = self.obs["elevation_map"][0].cpu().numpy()
        valid = values[elevation.cells:] > .5
        co, si = np.cos(pose[3]), np.sin(pose[3])
        xy = elevation.grid.cpu().numpy()[valid] @ np.array([[co, si], [-si, co]]) + pose[:2]
        self.map_points = np.column_stack((xy, values[:elevation.cells][valid] + pose[2]))

    @torch.no_grad()
    def step(self):
        env = self.env
        if self.last_fall:  # Headless runs still execute all N steps, like eval_loco.
            env.reset(env.ids)
            self.map_stamp = -1
            self.set_command()
        self.obs, _, _, extras = env.step(self.policy(self.obs))
        if not self.fell and bool(extras["valid"][0]):
            self.distance = max(self.distance, float((env.qpos[0, :2] - self.start).norm()))
        self.last_fall = bool(extras["falls"][0])
        self.fell |= self.last_fall
        self.steps += 1
        self.cache_map()

    def on_key(self, key):
        if key == KEY_SPACE:
            self.paused = not self.paused
        elif key == ord("M") and self.student:
            self.show_map = not self.show_map
        elif key == KEY_BACKSPACE:
            self.reset()
        elif key in (ord("["), ord("]"), ord("-"), ord("=")) or ord("1") <= key <= ord("7"):
            if key in (ord("["), ord("]")):
                self.kind = (self.kind + (1 if key == ord("]") else -1)) % len(TILE_TYPES)
            elif key in (ord("-"), ord("=")):
                self.level = int(np.clip(self.level + (1 if key == ord("=") else -1), 0, self.max_level))
            else:
                self.kind = key - ord("1")
            self.reset()
        elif key == KEY_HOME or key in COMMAND_KEYS:
            if key == KEY_HOME:
                self.cmd[:] = 0
            else:
                axis, delta = COMMAND_KEYS[key]
                self.cmd[axis] = np.clip(self.cmd[axis] + delta, CMD_LO[axis], CMD_HI[axis])
            self.set_command()

    def sync_model(self):
        self.vd.qpos[:] = self.env.qpos[0].cpu().numpy()
        self.vd.qvel[:] = self.env.qvel[0].cpu().numpy()
        self.vd.ctrl[:] = self.env.ctrl[0].cpu().numpy()
        self.vd.time = self.steps * self.env.dt
        mujoco.mj_forward(self.vm, self.vd)

    def sync_view(self, viewer):
        self.sync_model()
        viewer.cam.lookat[:] = self.vd.qpos[:3]
        scn = viewer.user_scn
        scn.ngeom = 0
        if self.show_map:
            for point in self.map_points[:scn.maxgeom]:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                   np.full(3, .015), point, np.eye(3).ravel(), np.array([.15, .9, .5, .8]))
                scn.ngeom += 1

    def overlay(self):
        rows = [("policy", self.label), ("terrain", f"{TILE_TYPES[self.kind]}  level {self.level}"),
                ("command", f"vx {self.cmd[0]:+.2f}  vy {self.cmd[1]:+.2f}  yaw {self.cmd[2]:+.2f}"),
                ("keys", "arrows/PgUp/PgDn steer  Home stop  Space pause  Backspace reset"),
                ("terrain keys", "1..7 / [ ] type   - = level   M map"),
                ("state", ("PAUSED  " if self.paused else "") + ("FELL (Backspace resets, Space resumes)" if self.fell else "walking"))]
        return (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "\n".join(k for k, _ in rows), "\n".join(v for _, v in rows))

    def run(self):
        import mujoco.viewer

        # The viewer callback runs on another thread; keep CUDA and resets here.
        with mujoco.viewer.launch_passive(self.vm, self.vd, key_callback=self.keys.put) as viewer:
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.6, 135., -20.
            viewer.opt.geomgroup[3] = 0
            deadline = time.perf_counter()
            while viewer.is_running():
                while not self.keys.empty():
                    self.on_key(self.keys.get())
                if not self.paused:
                    self.step()
                    if self.last_fall:
                        self.paused = True
                with viewer.lock():
                    self.sync_view(viewer)
                viewer.set_texts(self.overlay())  # Takes its own lock; keep outside viewer.lock().
                viewer.sync(state_only=True)  # The large terrain/model stays static.
                deadline += self.env.dt
                time.sleep(max(0., deadline - time.perf_counter()))
                if time.perf_counter() - deadline > .5:
                    deadline = time.perf_counter()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", type=Path)
    p.add_argument("--terrain", choices=TILE_TYPES, default="flat")
    p.add_argument("--level", type=int, choices=range(10), default=0)
    p.add_argument("--map_noise", choices=("on", "off"), default="on")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--headless", type=int, default=0, metavar="N", help="run N control steps without a window")
    args = p.parse_args()
    if args.headless < 0:
        p.error("--headless must be nonnegative")
    print("Keys" + __doc__.split("Keys", 1)[1], flush=True)
    print("Tiles: " + "  ".join(f"{k + 1}={name}" for k, name in enumerate(TILE_TYPES)), flush=True)
    setup_device(args.device)
    app = Player(args.ckpt, args.terrain, args.level, args.map_noise == "on", args.device)
    print(f"{app.label}: {args.terrain} level {args.level}, vx=+0.6 m/s, device={args.device}, "
          f"map_noise={args.map_noise if app.student else 'n/a'}, one GPU world at {1 / app.env.dt:g} Hz", flush=True)
    if args.headless:
        for _ in range(args.headless):
            app.step()
            app.sync_model()
        print(f"{app.steps} steps ({app.steps * app.env.dt:.2f} s): distance={app.distance:.3f} m "
              f"(max XY displacement before first fall), fell={app.fell}, blowups={int(app.env.total_blown)}", flush=True)
    else:
        app.run()


if __name__ == "__main__":
    main()
