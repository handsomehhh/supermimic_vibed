"""Check the simulator against the reference library: state round-trip, FK vs simulator body frames,
reward on open-loop reference playback, standing stability, and throughput.

  python tools/test_env.py --packs data/packs/lafan_loco.pt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--packs", nargs="+", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--bench_envs", type=int, default=4096)
args = parser.parse_args()

import torch
import warp as wp

from supermimic.env import TrackingSim, use_warp_stream
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP
from supermimic.motion import MotionLib
from supermimic.robot import TRACKED_BODIES, default_joint_pos_list
from supermimic.rotations import quat_to_mat, rot_angle_sq

dev = "cuda:0"
use_warp_stream(dev)
kin = G1Kinematics(dev)
t0 = time.time()
motion = MotionLib.from_packs(args.packs, args.packs, kin, dev)
print(f"motion: {motion.num_clips} clips, {motion.num_frames} frames ({motion.num_frames * 0.02 / 3600:.2f} h), bins {motion.num_bins}, load {time.time() - t0:.1f}s")
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
N = args.num_envs
env = TrackingSim(N, motion, mdp, dev, eval_mode=True)
frames = motion.clip_start[torch.arange(N, device=dev) % motion.num_clips] + 100
env.reset(torch.arange(N, device=dev), frames)
S = env.state()
for n, a, b in zip(["p", "R", "v", "w", "q", "qd"], S, motion.ref_state(frames)):
    print(f"roundtrip {n}: max err {(a - b).abs().max().item():.2e}")

# FK vs simulator body frames (after one zero-duration kinematics pass)
import mujoco_warp as mjw

with wp.ScopedDevice(env.wp_device):
    mjw.kinematics(env.model, env.data)
    mjw.com_pos(env.model, env.data)
    mjw.com_vel(env.model, env.data)
ids = [env.mjm.body(b).id for b in TRACKED_BODIES]
sp = wp.to_torch(env.data.xpos)[:, ids]
sR = quat_to_mat(wp.to_torch(env.data.xquat)[:, ids])
bp, bR, bv, bw = kin.forward(*S)
print(f"FK pos err {(bp - sp).norm(dim=-1).max().item():.2e}, rot err {rot_angle_sq(bR, sR).sqrt().max().item():.2e}")
print(f"reference body vs FK(sim) pos err {(motion.bodies(frames)[0] - bp).norm(dim=-1).max().item():.2e}")

# open-loop reference playback via joint targets
for t in range(150):
    a = (motion.q[torch.minimum(env.f + 1, motion.clip_end[motion.clip_id[env.f]])] - env.default_q) / env.alpha
    out = env.step(a)
    if t % 25 == 0:
        print(f"t {t}: norm reward {out['r'].mean().item() / 0.1:.3f}, terminated {env.failed_eval.float().mean().item():.2f}")

# standing with zero action: should stay up
env_v = TrackingSim(N, motion, mdp, dev, eval_mode=True, n_vel=N)
env_v.reset_standing(torch.arange(N, device=dev), motion.clip_start[:1].expand(N).clone())
for t in range(100):
    env_v.step(torch.zeros(N, 29, device=dev))
S = env_v.state()
print(f"standing 2 s, zero action: pelvis height {S[0][:, 2].mean().item():.3f} (start {env_v.stand_height:.3f}), fell {mdp.fall(S).float().mean().item():.2f}")

# throughput
env_b = TrackingSim(args.bench_envs, motion, mdp, dev)
a = torch.zeros(args.bench_envs, 29, device=dev)
for _ in range(10):
    env_b.step(a)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    env_b.step(0.3 * torch.randn_like(a))
torch.cuda.synchronize()
print(f"throughput: {100 * args.bench_envs / (time.time() - t0):.0f} control steps/s at {args.bench_envs} envs")
