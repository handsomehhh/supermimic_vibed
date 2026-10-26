"""Checkpoint evaluation of both goal interfaces.

Motion tracking (paper App. F.1): each evaluated motion plays once from its first frame with friction
1.0, no domain randomization, no observation or action noise. A motion fails if the rollout triggers
the training termination. Normalized return is the undiscounted reward sum over its maximum. At most
``--max_clips`` motion clips (a fixed random subset) and ``--max_loco_clips`` velocity-library clips
are evaluated.

Velocity commands (App. G.2): from standing, the policy receives a fixed command for 5 s; the relative
steady-state error is the norm of the mean (vx, vy, yaw rate) error over the last 2 s over the command
norm. 672 commands (15 x 15 grids on the three planes of the command space, standing left out), 4
trials each; the per-command value is the median over the trials that did not fall (0.9 if all fell).

Run as a watcher: ``python supermimic/evaluate.py --run_dir logs/<run> --watch``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--watch", action="store_true")
parser.add_argument("--ckpt", type=str, default=None, help="evaluate one checkpoint")
parser.add_argument("--every", type=int, default=1, help="evaluate every n-th checkpoint")
parser.add_argument("--max_clips", type=int, default=1024)
parser.add_argument("--max_loco_clips", type=int, default=512)
parser.add_argument("--trials", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

import torch

import supermimic.networks as networks
from supermimic.env import TrackingSim, use_warp_stream
from supermimic.env import CMD_HI, CMD_LO
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MAX_STEP_REWARD, MDP, OBS_DIM, REL_DIM
from supermimic.motion import MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list

dev = args.device
use_warp_stream(dev)
torch.backends.cuda.matmul.allow_tf32 = True
while not os.path.exists(os.path.join(args.run_dir, "args.json")):  # the trainer writes it at start-up
    time.sleep(10)
run_args = json.load(open(os.path.join(args.run_dir, "args.json")))
kin = G1Kinematics(dev)
motion = MotionLib.from_packs(run_args["motion_packs"], run_args["loco_packs"], kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
VEL = len(run_args["loco_packs"]) > 0
K, Hh = run_args["goal_frames"], run_args["hist"]

gen = torch.Generator().manual_seed(0)


def subset(mask: torch.Tensor, n: int) -> torch.Tensor:
    ids = mask.nonzero(as_tuple=False).squeeze(-1).cpu()
    return ids if len(ids) <= n else ids[torch.randperm(len(ids), generator=gen)[:n]].sort().values


mo_ids = subset(motion.in_motion, args.max_clips)
lo_ids = subset(motion.in_loco, args.max_loco_clips) if VEL else torch.zeros(0, dtype=torch.long)
clip_ids = torch.cat((mo_ids, lo_ids)).to(dev)
n_mo = len(mo_ids)
env_m = TrackingSim(len(clip_ids), motion, mdp, dev, eval_mode=True)


def step_test_commands():
    lo, hi = CMD_LO, CMD_HI
    cmds = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        ga, gb = torch.linspace(lo[a], hi[a], 15), torch.linspace(lo[b], hi[b], 15)
        A, B = torch.meshgrid(ga, gb, indexing="ij")
        c = torch.zeros(225, 3)
        c[:, a], c[:, b] = A.flatten(), B.flatten()
        cmds.append(c[c.norm(dim=-1) > 1e-6])
    return torch.cat(cmds)


if VEL:
    CMDS = step_test_commands().to(dev)
    env_v = TrackingSim(len(CMDS) * args.trials, motion, mdp, dev, eval_mode=True, n_vel=len(CMDS) * args.trials)
    longest = motion.clip_start[torch.argmax(torch.where(motion.in_loco, motion.clip_len, 0))]


def load_policy(ck: dict) -> Policy:
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"].get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, 73 * K + REL_DIM, hist=Hh).to(dev)
    pi.load_state_dict(ck["pi"])
    return pi.eval()


@torch.no_grad()
def eval_motion(pi: Policy) -> dict:
    env = env_m
    C = len(clip_ids)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset(torch.arange(C, device=dev), motion.clip_start[clip_ids].clone())
    ret = torch.zeros(C, device=dev)
    hist = None
    while not bool(env.done_eval.all()):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, mdp.motion_goal(S[1], env.f, K, noise=False))
        alive = ~env.done_eval.clone()
        out = env.step(a)
        ret += torch.where(alive, out["r"], 0.0)
    norm = ret / ((motion.clip_len[clip_ids] - 1).float() * MAX_STEP_REWARD)
    failed = env.failed_eval
    res = {"norm_return": norm[:n_mo].mean().item(), "n_clips": n_mo, "n_failed": int(failed[:n_mo].sum()),
           "fail_rate": failed[:n_mo].float().mean().item(),
           "failed": [motion.names[i] for i in clip_ids[:n_mo][failed[:n_mo]].tolist()][:50]}  # fmt: skip
    if len(lo_ids):
        res.update(loco_norm_return=norm[n_mo:].mean().item(), loco_fail_rate=failed[n_mo:].float().mean().item())
    return res


@torch.no_grad()
def eval_velocity(pi: Policy) -> dict:
    env = env_v
    T, n = len(CMDS), env.N
    ids = torch.arange(n, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset_standing(ids, longest.expand(n).clone())
    env.cmd[:] = CMDS.repeat(args.trials, 1)
    env.cmd_ref[:] = False
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    err_sum = torch.zeros(n, 3, device=dev)
    hist = None
    for t in range(250):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        env.step(pi.act_vel(hist, mdp.norm_cmd(env.cmd)))
        S1 = env.state()
        fell |= mdp.fall(S1)
        if t >= 150:
            err_sum += mdp.heading_vel(S1) - env.cmd
    rel = ((err_sum / 100).norm(dim=-1) / env.cmd.norm(dim=-1)).view(args.trials, T)
    fell = fell.view(args.trials, T)
    per_cmd = torch.where(fell, torch.nan, rel).nanmedian(0).values
    all_fell = fell.all(0)
    per_cmd = torch.where(all_fell, 0.9, per_cmd.clamp(max=0.9))
    return {"vel_med_err": per_cmd.median().item(), "vel_mean_err": per_cmd.mean().item(), "vel_fall_rate": fell.float().mean().item(),
            "vel_cmds_all_fell": int(all_fell.sum()), "vel_per_cmd": [round(x, 4) for x in per_cmd.tolist()]}  # fmt: skip


def evaluate(ck: dict) -> dict:
    pi = load_policy(ck)
    res = eval_motion(pi)
    if VEL:
        res.update(eval_velocity(pi))
    return res


COLS = ["iteration", "train_wall_h", "saved_at", "eval_at", "norm_return", "fail_rate", "n_failed", "n_clips", "loco_norm_return",
        "loco_fail_rate", "vel_med_err", "vel_mean_err", "vel_fall_rate", "vel_cmds_all_fell"]  # fmt: skip

if args.ckpt:
    res = evaluate(torch.load(args.ckpt, map_location=dev))
    print(json.dumps({k: v for k, v in res.items() if k != "vel_per_cmd"}, indent=1))
    sys.exit(0)
out_csv = os.path.join(args.run_dir, "eval.csv")
done = set()
if os.path.exists(out_csv):
    done = {int(line.split(",")[0]) for line in open(out_csv).read().strip().splitlines()[1:]}
else:
    open(out_csv, "w").write(",".join(COLS) + "\n")
while True:
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
    todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
    todo = [c for c in todo if (int(os.path.basename(c)[5:11]) // 2000) % args.every == 0]
    for c in todo[-1:] if args.watch else todo:  # a watcher that falls behind evaluates the newest checkpoint
        it = int(os.path.basename(c)[5:11])
        ck = torch.load(c, map_location=dev)
        t0 = time.time()
        res = evaluate(ck)
        row = {"iteration": it, "train_wall_h": f"{ck['wall_time'] / 3600:.4f}", "saved_at": ck["saved_at"],
               "eval_at": time.strftime("%Y-%m-%d %H:%M:%S"), **res}  # fmt: skip
        open(out_csv, "a").write(",".join(str(row.get(k, "")) if not isinstance(row.get(k), float) else f"{row[k]:.4f}" for k in COLS) + "\n")
        json.dump(res, open(c.replace(".pt", "_eval.json"), "w"), indent=1)
        print(f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): motion return {res['norm_return']:.4f}, failed {res['n_failed']}/{res['n_clips']}"
              + (f" | vel med err {res['vel_med_err']:.3f}, fall {res['vel_fall_rate']:.3f}" if VEL else "")
              + f" ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
        done.add(it)
    if not args.watch:
        break
    time.sleep(60)
