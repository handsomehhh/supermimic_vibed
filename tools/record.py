"""Record rollouts of a checkpoint for replay (tools/play.py, tools/render.py).

Motion tracking: the chosen clips play from their first frame (evaluation settings: friction 1.0, no
randomization, no noise), with the reference frame of every step. Velocity commands: from standing, a
scripted tour of the command space plus random command sequences; each step keeps the command and the
measured (vx, vy, yaw rate).

  python tools/record.py --ckpt logs/exp3/s6/ckpt_033950.pt --out rec/exp3.npz --n_motion 12 --n_loco 6 --failed 6
  python tools/record.py --ckpt ... --out ... --clips walk1_subject1 dance2   # name substrings
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--label", default="", help="shown in the player, e.g. 'exp3 s6 it 33950'")
ap.add_argument("--clips", nargs="*", default=[], help="clips whose name contains one of these")
ap.add_argument("--n_motion", type=int, default=12, help="random motion-tracking clips")
ap.add_argument("--n_loco", type=int, default=6, help="random clips of the velocity library")
ap.add_argument("--failed", type=int, default=6, help="clips the checkpoint's evaluation failed (from <ckpt>_eval.json)")
ap.add_argument("--max_seconds", type=float, default=30.0)
ap.add_argument("--n_random_cmd", type=int, default=5, help="random velocity-command sequences besides the tour")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import numpy as np
import torch

import supermimic.networks as networks
from supermimic.env import CMD_HI, CMD_LO, TrackingSim, use_warp_stream
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MAX_STEP_REWARD, MDP, OBS_DIM, REL_DIM
from supermimic.motion import CONTROL_DT, MotionLib
from supermimic.networks import Policy
from supermimic.robot import default_joint_pos_list
from supermimic.rotations import mat_to_quat

dev = args.device
use_warp_stream(dev)
ck = torch.load(args.ckpt, map_location=dev)
ra = ck["args"]
K, Hh = ra["goal_frames"], ra["hist"]
kin = G1Kinematics(dev)
motion = MotionLib.from_packs(ra["motion_packs"], ra.get("loco_packs", []), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
networks.AMP_DTYPE = torch.bfloat16 if ra.get("amp", "none") == "bf16" else None
pi = Policy(OBS_DIM, 73 * K + REL_DIM, hist=Hh).to(dev)
pi.load_state_dict(ck["pi"])
pi.eval()
VEL = len(ra.get("loco_packs", [])) > 0
gen = torch.Generator().manual_seed(args.seed)
T_MAX = int(args.max_seconds / CONTROL_DT)


def pick(mask: torch.Tensor, n: int) -> list[int]:
    ids = mask.nonzero(as_tuple=False).squeeze(-1).cpu()
    return ids[torch.randperm(len(ids), generator=gen)[:n]].tolist() if n > 0 else []


# ------------------------------------------------------------------ which clips
name_to_id = {n: i for i, n in enumerate(motion.names)}
chosen, why = [], []


def add(ids, tag):
    for i in ids:
        if i not in chosen:
            chosen.append(i), why.append(tag)


for s in args.clips:
    add([i for i, n in enumerate(motion.names) if s in n][:4], "named")
eval_json = args.ckpt.replace(".pt", "_eval.json")
if args.failed and os.path.exists(eval_json):
    add([name_to_id[n] for n in json.load(open(eval_json)).get("failed", []) if n in name_to_id][: args.failed], "failed in eval")
add(pick(motion.in_motion & ~motion.in_loco if bool((motion.in_motion & ~motion.in_loco).any()) else motion.in_motion, args.n_motion), "random")
if VEL:
    add(pick(motion.in_loco, args.n_loco), "locomotion")
C = len(chosen)
cid = torch.tensor(chosen, device=dev)
print(f"[record] {C} clips: " + ", ".join(f"{motion.names[i]} ({w})" for i, w in zip(chosen, why)), flush=True)


@torch.no_grad()
def record_tracking():
    env = TrackingSim(C, motion, mdp, dev, eval_mode=True)
    env.reset(torch.arange(C, device=dev), motion.clip_start[cid].clone())
    T = int(min(T_MAX, int(motion.clip_len[cid].max()) - 1))
    rec = {k: [] for k in ("root_p", "root_quat", "q", "ref_root_p", "ref_root_quat", "ref_q", "alive", "t")}
    ret = torch.zeros(C, device=dev)
    hist = None
    for _ in range(T):
        S = env.state()
        f = env.f.clone()
        rec["root_p"].append(S[0]), rec["root_quat"].append(mat_to_quat(S[1])), rec["q"].append(S[4])
        rec["ref_root_p"].append(motion.root_p[f]), rec["ref_root_quat"].append(mat_to_quat(motion.root_R[f])), rec["ref_q"].append(motion.q[f])
        rec["alive"].append(~env.done_eval), rec["t"].append((f - motion.clip_start[cid]).float() * CONTROL_DT)
        o = mdp.obs(S, env.a_prev, env.xi)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        alive = ~env.done_eval.clone()
        out = env.step(pi(hist, mdp.motion_goal(S[1], env.f, K, noise=False)))
        ret += torch.where(alive, out["r"], 0.0)
    steps = torch.minimum(motion.clip_len[cid] - 1, torch.tensor(T, device=dev)).float()
    out = {f"trk_{k}": torch.stack(v).cpu().numpy() for k, v in rec.items()}
    out.update(trk_names=np.array([motion.names[i] for i in chosen]), trk_why=np.array(why),
               trk_failed=env.failed_eval.cpu().numpy(), trk_return=(ret / (steps * MAX_STEP_REWARD)).cpu().numpy(),
               trk_len=steps.long().cpu().numpy())  # fmt: skip
    return out


# ------------------------------------------------------------------ velocity commands
TOUR = [  # (seconds, vx, vy, yaw rate, label)
    (2, 0, 0, 0, "stand"), (3, 0.6, 0, 0, "walk forward"), (3, 1.2, 0, 0, "walk faster"), (3, 2.0, 0, 0, "run"),
    (2, 0, 0, 0, "stop"), (3, -1.0, 0, 0, "walk backward"), (3, 0, 0.8, 0, "side step left"), (3, 0, -0.8, 0, "side step right"),
    (3, 0.8, 0, 1.2, "curve left"), (3, 0, 0, 2.5, "turn in place"), (3, 1.0, 0.5, -1.0, "diagonal + turn"), (2, 0, 0, 0, "stop"),
]  # fmt: skip


def program(i: int, T: int) -> tuple[np.ndarray, list[str]]:
    """Per-step command (T, 3) and its label for program i (0: the tour, else random 3 s segments)."""
    cmd, lab = np.zeros((T, 3), np.float32), [""] * T
    if i == 0:
        t = 0
        for sec, vx, vy, wz, name in TOUR:
            n = int(sec / CONTROL_DT)
            cmd[t : t + n], lab[t : t + n] = (vx, vy, wz), [name] * n
            t += n
        cmd[t:], lab[t:] = 0, ["stop"] * (T - t)
        return cmd, lab
    rng = np.random.default_rng(args.seed * 100 + i)
    lo, hi = np.array(CMD_LO), np.array(CMD_HI)
    for t0 in range(0, T, int(3 / CONTROL_DT)):
        c = lo + (hi - lo) * rng.random(3)
        if rng.random() < 0.15:
            c[:] = 0
        cmd[t0 : t0 + int(3 / CONTROL_DT)] = c
        lab[t0 : t0 + int(3 / CONTROL_DT)] = ["random command"] * len(lab[t0 : t0 + int(3 / CONTROL_DT)])
    return cmd, lab


@torch.no_grad()
def record_velocity():
    E = 1 + args.n_random_cmd
    T = int(sum(s for s, *_ in TOUR) / CONTROL_DT) + 100
    progs = [program(i, T) for i in range(E)]
    cmds = torch.tensor(np.stack([p[0] for p in progs], 1), device=dev)  # (T, E, 3)
    env = TrackingSim(E, motion, mdp, dev, eval_mode=True, n_vel=E)
    longest = motion.clip_start[torch.argmax(torch.where(motion.in_loco, motion.clip_len, 0))]
    env.reset_standing(torch.arange(E, device=dev), longest.expand(E).clone())
    env.cmd_ref[:] = False
    rec = {k: [] for k in ("root_p", "root_quat", "q", "cmd", "meas", "alive")}
    fell = torch.zeros(E, dtype=torch.bool, device=dev)
    hist = None
    for t in range(T):
        env.cmd[:] = cmds[t]
        S = env.state()
        rec["root_p"].append(S[0]), rec["root_quat"].append(mat_to_quat(S[1])), rec["q"].append(S[4])
        rec["cmd"].append(cmds[t]), rec["meas"].append(mdp.heading_vel(S)), rec["alive"].append(~fell)
        o = mdp.obs(S, env.a_prev, env.xi)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        env.step(pi.act_vel(hist, mdp.norm_cmd(env.cmd)))
        fell |= mdp.fall(env.state())
    out = {f"vel_{k}": torch.stack(v).cpu().numpy() for k, v in rec.items()}
    out.update(vel_names=np.array(["command tour"] + [f"random commands {i}" for i in range(1, E)]),
               vel_label=np.array([p[1] for p in progs]).T)  # (T, E)  # fmt: skip
    return out


data = record_tracking()
if VEL:
    data.update(record_velocity())
label = args.label or f"{os.path.basename(os.path.dirname(args.ckpt))} it {ck['iteration']}"
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez_compressed(args.out, label=np.array(label), ckpt=np.array(args.ckpt), dt=np.array(CONTROL_DT), **data)
msg = f"[record] {label}: {C} clips, mean return {data['trk_return'].mean():.3f}, failed {int(data['trk_failed'].sum())}"
if VEL:
    msg += f"; {len(data['vel_names'])} command sequences, fell {int((~data['vel_alive'][-1]).sum())}"
print(msg + f" -> {args.out}", flush=True)
