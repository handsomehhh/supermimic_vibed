"""Checkpoint evaluation of both goal interfaces.

Motion tracking (paper App. F.1): each evaluated motion plays once from its first frame with friction
1.0, no domain randomization, no observation or action noise. A motion fails if the rollout triggers
the training termination. Normalized return is the undiscounted reward sum over its maximum. At most
``--max_clips`` motion clips (a fixed random subset) and ``--max_loco_clips`` velocity-library clips
are evaluated.

Velocity commands (App. G.2): from standing, the policy receives a fixed command for 5 s; the relative
steady-state error is the norm of the mean (vx, vy, yaw rate) error over the last 2 s over the command
norm. 672 commands (15 x 15 grids on the three planes of the command space, standing left out), 4
trials each; the per-command value is the median over the trials that did not fall (0.9 if all fell). Runs with ``upper_cmd``: the first half of the trials leave the
waist and arms untracked (comparable to runs without it), the second half track the waist/arm pose of a
fixed random motion-library frame per command; ``upper_med_err`` is the median over those non-falling
trials of the mean |joint error| (rad) over the last 2 s, and the velocity error is also reported per half.
Runs with ``terrain``: all simulators are the terrain scene (motion clips on the floor around the origin, the
step test on the floor with the head camera), plus a traversal test: on every tile type x 10 levels x 4 trials,
from the tile center facing +x, a 0.8 m/s forward command for 10 s; success = 3.5 m out (past the last step or
the slope) without falling. Raised pyramids (stairs_up, slope_up) test descending, pits test ascending.

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
from supermimic.mdp import MAX_STEP_REWARD, MDP, OBS_DIM, REL_DIM, UPPER, UPPER_GROUPS
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
UPPER_CMD = run_args.get("upper_cmd", False)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev), upper_cmd=UPPER_CMD)
VEL = len(run_args["loco_packs"]) > 0
K, Hh = run_args["goal_frames"], run_args["hist"]
SELF_COL = run_args.get("self_collision", False)  # runs before the flag existed had none

gen = torch.Generator().manual_seed(0)


def subset(mask: torch.Tensor, n: int) -> torch.Tensor:
    ids = mask.nonzero(as_tuple=False).squeeze(-1).cpu()
    return ids if len(ids) <= n else ids[torch.randperm(len(ids), generator=gen)[:n]].sort().values


mo_ids = subset(motion.in_motion, args.max_clips)
lo_ids = subset(motion.in_loco, args.max_loco_clips) if VEL else torch.zeros(0, dtype=torch.long)
clip_ids = torch.cat((mo_ids, lo_ids)).to(dev)
n_mo = len(mo_ids)
TERRAIN = run_args.get("terrain", False) and VEL
if TERRAIN:
    from supermimic.terrain import TILE_TYPES
    from supermimic.terrain_env import TerrainSetup, TerrainSim

    SETUP = TerrainSetup(seed=run_args.get("terrain_seed", 0))
    Sim = lambda n, n_vel=0: TerrainSim(n, motion, mdp, dev, eval_mode=True, n_vel=n_vel, self_collision=SELF_COL, setup=SETUP)  # noqa: E731
else:
    Sim = lambda n, n_vel=0: TrackingSim(n, motion, mdp, dev, eval_mode=True, n_vel=n_vel, self_collision=SELF_COL)  # noqa: E731
env_m = Sim(len(clip_ids))


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
    env_v = Sim(len(CMDS) * args.trials, len(CMDS) * args.trials)
    longest = motion.clip_start[torch.argmax(torch.where(motion.in_loco, motion.clip_len, 0))]
    n_tr = len(CMDS) * args.trials
    CMDS_FULL = mdp.full_cmd(CMDS.repeat(args.trials, 1))  # (trials * commands, cmd_dim), trial-major
    UP_TRIAL = torch.zeros(n_tr, dtype=torch.bool, device=dev)
    if UPPER_CMD:
        UP_TRIAL = (torch.arange(n_tr, device=dev) // len(CMDS)) >= args.trials // 2
        mo_frames = torch.cat([torch.arange(int(motion.clip_start[c]), int(motion.clip_end[c]) + 1, 25) for c in mo_ids.tolist()])
        pick = mo_frames[torch.randint(len(mo_frames), (n_tr,), generator=gen)].to(dev)
        track = UP_TRIAL.float().unsqueeze(-1).expand(n_tr, len(UPPER_GROUPS))
        CMDS_FULL = mdp.full_cmd(CMDS.repeat(args.trials, 1), motion.q[pick][:, UPPER], track)


def load_policy(ck: dict) -> Policy:
    networks.AMP_DTYPE = torch.bfloat16 if ck["args"].get("amp", "none") == "bf16" else None
    pi = Policy(OBS_DIM, 73 * K + REL_DIM, hist=Hh, cmd_dim=mdp.cmd_dim, depth_frames=SETUP.camera.frames if TERRAIN else 0).to(dev)
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
    env.cmd[:] = CMDS_FULL
    env.cmd_ref[:] = False
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    err_sum = torch.zeros(n, 3, device=dev)
    up_sum = torch.zeros(n, device=dev)
    hist = None
    for t in range(250):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        env.step(pi.act_vel(hist, mdp.norm_cmd(env.cmd), env.depth_stack.float() if TERRAIN else None))
        S1 = env.state()
        fell |= mdp.fall(S1)
        if t >= 150:
            err_sum += mdp.heading_vel(S1) - env.cmd[:, :3]
            if UPPER_CMD:
                up_sum += mdp.upper_err(S1[4], env.cmd)[1]
    rel = ((err_sum / 100).norm(dim=-1) / env.cmd[:, :3].norm(dim=-1)).view(args.trials, T)
    fell = fell.view(args.trials, T)

    def per_command(rows: slice):
        per = torch.where(fell[rows], torch.nan, rel[rows]).nanmedian(0).values
        all_fell = fell[rows].all(0)
        return torch.where(all_fell, 0.9, per.clamp(max=0.9)), all_fell

    per_cmd, all_fell = per_command(slice(None))
    res = {"vel_med_err": per_cmd.median().item(), "vel_mean_err": per_cmd.mean().item(), "vel_fall_rate": fell.float().mean().item(),
           "vel_cmds_all_fell": int(all_fell.sum()), "vel_per_cmd": [round(x, 4) for x in per_cmd.tolist()]}  # fmt: skip
    if UPPER_CMD:
        h = args.trials // 2
        up = (up_sum / 100).view(args.trials, T)[h:]
        res.update(upper_med_err=up[~fell[h:]].median().item() if bool((~fell[h:]).any()) else float("nan"),
                   vel_med_err_free=per_command(slice(0, h))[0].median().item(),
                   vel_med_err_up=per_command(slice(h, None))[0].median().item(),
                   vel_fall_rate_up=fell[h:].float().mean().item())  # fmt: skip
    return res


if TERRAIN:
    _cols = env_v.kinds_map[0]
    T_KINDS = [k for k in range(len(TILE_TYPES)) if bool((_cols == k).any())]
    T_LEVELS, T_TRIALS = env_v.origins.shape[0], 4
    n_t = len(T_KINDS) * T_LEVELS * T_TRIALS
    env_t = Sim(n_t, n_t)
    _k = torch.tensor(T_KINDS, device=dev).repeat_interleave(T_LEVELS * T_TRIALS)
    T_COL = torch.stack([(_cols == k).nonzero()[0, 0] for k in _k.tolist()])
    T_LEVEL = torch.arange(T_LEVELS, device=dev).repeat_interleave(T_TRIALS).repeat(len(T_KINDS))


@torch.no_grad()
def eval_terrain(pi: Policy) -> dict:
    env = env_t
    n = env.N
    ids = torch.arange(n, device=dev)
    env.done_eval[:] = False
    env.failed_eval[:] = False
    env.reset_on_tile(ids, T_COL, T_LEVEL, torch.zeros(n, device=dev), longest.expand(n).clone())
    env.cmd[:] = mdp.full_cmd(torch.tensor([[0.8, 0.0, 0.0]], device=dev).expand(n, 3))
    env.cmd_ref[:] = False
    start = env.qpos[:, :2].clone()
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    reached = torch.zeros(n, dtype=torch.bool, device=dev)
    hist = None
    for t in range(500):
        S = env.state()
        o = mdp.obs(S, env.a_prev, env.xi)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        env.step(pi.act_vel(hist, mdp.norm_cmd(env.cmd), env.depth_stack.float()))
        S1 = env.state()
        fell |= env.task.fall(S1) & ~reached
        reached |= ((S1[0][:, :2] - start).norm(dim=-1) >= 3.5) & ~fell
    ok = (reached & ~fell).float().view(len(T_KINDS), T_LEVELS, T_TRIALS)
    res = {"terrain_success": ok.mean().item()}
    for i, k in enumerate(T_KINDS):
        res[f"terrain_{TILE_TYPES[k]}"] = ok[i].mean().item()
        res[f"terrain_{TILE_TYPES[k]}_levels"] = [round(x, 2) for x in ok[i].mean(-1).tolist()]
    return res


def evaluate(ck: dict) -> dict:
    pi = load_policy(ck)
    res = eval_motion(pi)
    if VEL:
        res.update(eval_velocity(pi))
    if TERRAIN:
        res.update(eval_terrain(pi))
    return res


COLS = ["iteration", "train_wall_h", "saved_at", "eval_at", "norm_return", "fail_rate", "n_failed", "n_clips", "loco_norm_return",
        "loco_fail_rate", "vel_med_err", "vel_mean_err", "vel_fall_rate", "vel_cmds_all_fell"]  # fmt: skip
if UPPER_CMD:
    COLS += ["upper_med_err", "vel_med_err_free", "vel_med_err_up", "vel_fall_rate_up"]
if TERRAIN:
    COLS += ["terrain_success"] + [f"terrain_{TILE_TYPES[k]}" for k in T_KINDS]

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
              + (f" | upper err {res['upper_med_err']:.3f} rad, vel err free/upper {res['vel_med_err_free']:.3f}/{res['vel_med_err_up']:.3f}"
                 if VEL and UPPER_CMD else "")
              + (" | terrain " + ", ".join(f"{TILE_TYPES[k]} {res[f'terrain_{TILE_TYPES[k]}']:.2f}" for k in T_KINDS) if TERRAIN else "")
              + f" ({time.time() - t0:.0f}s)", flush=True)  # fmt: skip
        done.add(it)
    if not args.watch:
        break
    time.sleep(60)
