"""SuperMimic training (Algorithm A.1): collect -> fit world model -> BPTT policy + critic update.

Two goal interfaces train jointly through one world model (paper App. B, G): motion tracking on the
``--motion_packs`` library and, if ``--loco_packs`` is given, a velocity-command encoder trained by
RL on its own reward, with a stop-gradient on the shared decoder and a latent-alignment loss.
Multi-GPU: ``torchrun --nproc_per_node G supermimic/train.py ...`` (data parallel, gradients averaged).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096, help="per GPU")
parser.add_argument("--iterations", type=int, default=100_000)
parser.add_argument("--buffer_steps", type=int, default=1000)
parser.add_argument("--utd", type=float, default=0.4, help="updates per vectorized simulator step")
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--goal_frames", type=int, default=10)
parser.add_argument("--hist", type=int, default=10)
parser.add_argument("--wm_window", type=int, default=8)
parser.add_argument("--wm_batches", type=int, default=10)
parser.add_argument("--wm_batch", type=int, default=2048)
parser.add_argument("--pi_batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=2e-4, help="single-GPU learning rate; scaled by sqrt(min(#GPUs, --lr_gpu_cap))")
parser.add_argument("--lr_gpu_cap", type=int, default=2, help="the paper validates the sqrt scaling up to 2 GPUs; 8 GPUs at 5.7e-4 diverged")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--lam", type=float, default=0.95)
parser.add_argument("--action_noise", type=float, default=0.1)
parser.add_argument("--critic_epochs", type=int, default=5)
parser.add_argument("--critic_minibatches", type=int, default=4)
parser.add_argument("--warmup_steps", type=int, default=64, help="sim steps before any update")
parser.add_argument("--wm_only_iters", type=int, default=500, help="world-model-only iterations at the start")
parser.add_argument("--lr_decay_iters", type=int, default=250_000)
parser.add_argument("--ckpt_every", type=int, default=2000)
parser.add_argument("--log_every", type=int, default=50)
parser.add_argument("--motion_packs", nargs="+", required=True)
parser.add_argument("--loco_packs", nargs="*", default=[], help="velocity-interface library; empty: motion tracking only")
parser.add_argument("--vel_frac", type=float, default=0.5, help="share of envs and imagined rollouts on the velocity interface")
parser.add_argument("--align_w", type=float, default=0.1)
parser.add_argument("--log_dir", type=str, default=os.path.join(REPO, "logs", "run"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--amp", type=str, default="bf16", choices=["none", "bf16"], help="autocast dtype for MLP matmuls")
args = parser.parse_args()

import torch
import torch.distributed as dist
import torch.nn as nn

from supermimic.buffer import ReplayBuffer
from supermimic.env import TrackingSim, use_warp_stream
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import CRITIC_EXTRA, INV_DIM, MAX_STEP_REWARD, MAX_STEP_REWARD_VEL, MDP, OBS_DIM, REL_DIM, VEL_CRITIC_DIM
from supermimic.mdp import normalize_xi, unflatten_state
from supermimic.motion import MotionLib
import supermimic.networks as networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot import default_joint_pos_list

WORLD = int(os.environ.get("WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
dev = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
use_warp_stream(dev)
if WORLD > 1:
    dist.init_process_group("nccl", device_id=torch.device(dev))
MAIN = RANK == 0
LR = args.lr * math.sqrt(min(WORLD, args.lr_gpu_cap))

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(args.seed)  # identical initial weights on every rank
networks.AMP_DTYPE = torch.bfloat16 if args.amp == "bf16" else None
os.makedirs(args.log_dir, exist_ok=True)
if MAIN:
    json.dump({**vars(args), "world_size": WORLD, "lr_scaled": LR}, open(os.path.join(args.log_dir, "args.json"), "w"), indent=2)

kin = G1Kinematics(dev)
motion = MotionLib.from_packs(args.motion_packs, args.loco_packs, kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
VEL = len(args.loco_packs) > 0
K, Hh, HZ = args.goal_frames, args.hist, args.horizon
GOAL_DIM = 73 * K + REL_DIM
CRITIC_IN = INV_DIM + CRITIC_EXTRA + 73 * K
N_VEL = int(round(args.num_envs * args.vel_frac)) if VEL else 0
N_MO = args.num_envs - N_VEL
BV = int(round(args.pi_batch * args.vel_frac)) if VEL else 0
BM = args.pi_batch - BV
if MAIN:
    print(f"[motion] {motion.num_clips} clips, {motion.num_frames / 50 / 3600:.2f} h; motion clips {int(motion.in_motion.sum())}, "
          f"loco clips {int(motion.in_loco.sum())}; body cache {motion.cached_bodies}; world {WORLD}, lr {LR:.2e}", flush=True)  # fmt: skip

wm = WorldModel().to(dev)
pi = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
critic = Critic(CRITIC_IN).to(dev)
critic_v = Critic(VEL_CRITIC_DIM).to(dev)
opt_wm = torch.optim.Adam(wm.parameters(), lr=LR)
opt_pi = torch.optim.Adam(pi.parameters(), lr=LR, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=LR)
opt_vv = torch.optim.Adam(critic_v.parameters(), lr=LR)
MODELS = {"wm": wm, "pi": pi, "critic": critic, "critic_v": critic_v}
OPTS = {"opt_wm": opt_wm, "opt_pi": opt_pi, "opt_v": opt_v, "opt_vv": opt_vv}
start_it, wall_offset, sim_steps = 0, 0.0, 0
if args.resume:
    ck = torch.load(args.resume, map_location=dev)
    for k, m in {**MODELS, **OPTS}.items():
        m.load_state_dict(ck[k])
    start_it, wall_offset, sim_steps = ck["iteration"], ck["wall_time"], ck["sim_steps"]
if WORLD > 1:
    for m in MODELS.values():
        for t in list(m.parameters()) + list(m.buffers()):
            dist.broadcast(t.data, 0)
torch.manual_seed(args.seed + 1000 * RANK + start_it)  # independent environments per rank


def allreduce_grads(params):
    if WORLD == 1:
        return
    ps = list(params)
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in ps]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= WORLD
    for p, g in zip(ps, torch._utils._unflatten_dense_tensors(flat, grads)):
        p.grad = g


def sync_norms(*norms):
    """Average the running normalizer statistics across ranks (each rank sees the same amount of data)."""
    if WORLD == 1:
        return
    flat = torch.cat([torch.cat((n.mean, n.var)) for n in norms])
    dist.all_reduce(flat)
    flat /= WORLD
    i = 0
    for n in norms:
        d = n.mean.numel()
        n.mean.copy_(flat[i : i + d]), n.var.copy_(flat[i + d : i + 2 * d])
        i += 2 * d


# per-component floors on the error normalization (pos, rot, lin vel, ang vel, q, qd increments)
ERR_FLOOR = torch.cat((torch.full((3,), 3e-3), torch.full((3,), 5e-3), torch.full((3,), 2e-2), torch.full((3,), 5e-2),
                       torch.full((29,), 3e-3), torch.full((29,), 5e-2))).to(dev)  # fmt: skip


def _wm_rollout(s, a, xi):
    S = unflatten_state(s[:, 0])
    preds = []
    for j in range(args.wm_window):
        S = wm.step(S, a[:, j], normalize_xi(xi[:, j]))
        preds.append(S)
    return tuple(torch.stack([P[i] for P in preds], 1) for i in range(6))  # (B, k, ...)


def _wm_err(S_hat, s1, mask):
    std = torch.maximum(wm.out_norm.std(), ERR_FLOOR)
    e = (wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2  # (B, k, D)
    e = e.mean(-1)
    return (e * mask).sum() / s1.shape[0], e[:, -1]


def _imag_step(S, o_hist, f, xi, xi_n, cmd, cmd_ref):
    """One imagined step; rows [0, BM) use the motion interface, rows [BM, B) the velocity interface."""
    g = mdp.motion_goal(S[1], f, K)
    a = pi.decode(pi.z_motion(g[:BM]), o_hist[:BM])
    align = torch.zeros((), device=f.device)
    if VEL:
        zv = pi.z_vel(o_hist[BM:], mdp.norm_cmd(cmd[BM:]))
        av = pi.decode(zv, o_hist[BM:], frozen=True)
        with torch.no_grad():
            zt = pi.z_motion(g[BM:])
        m = cmd_ref[BM:].float()
        align = (((zv - zt) ** 2).mean(-1) * m).sum() / m.sum().clamp(min=1)
        a = torch.cat((a, av), 0)
    S1 = wm.step(S, a, xi_n)
    f1 = torch.minimum(f + 1, motion.clip_end[motion.clip_id[f]])
    o1 = mdp.add_noise(mdp.obs(S1, a, xi))
    o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
    cmd1 = torch.where(cmd_ref.unsqueeze(-1), motion.ref_cmd[f1], cmd)
    return S1, a, o_hist, f1, cmd1, align


def _reward_term(S, f, a, a_prev):
    return mdp.reward(S, f, a, a_prev), mdp.soft_termination(S, f)


def _vel_reward_term(S, cmd, a, a_prev):
    return mdp.vel_reward(S, cmd, a, a_prev), mdp.soft_fall(S)


def _critic_input(S, f, xi_n):
    return torch.cat((mdp.critic_features(S, f, xi_n), motion.goal(f, K)), -1)


def _vel_critic_input(S, cmd, xi_n):
    return mdp.vel_critic_features(S, cmd, xi_n)


COMPILE = not int(os.environ.get("SM_NO_COMPILE", "0"))
cc = torch.compile if COMPILE else (lambda fn: fn)
wm_rollout, wm_err, imag_step = cc(_wm_rollout), cc(_wm_err), cc(_imag_step)
reward_term, vel_reward_term = cc(_reward_term), cc(_vel_reward_term)
critic_input_c, vel_critic_input_c = cc(_critic_input), cc(_vel_critic_input)


def lr_at(it: int) -> float:
    if it >= args.lr_decay_iters:
        return LR * 0.03
    return LR * (0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / args.lr_decay_iters)))


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


# ------------------------------------------------------------------------------------------------
def wm_update():
    w = buffer.sample_windows(args.wm_batch, args.wm_window)
    S_hat = wm_rollout(w["s"], w["a"], w["xi"])
    loss, e_last = wm_err(S_hat, w["s1"], w["mask"])
    opt_wm.zero_grad(set_to_none=True)
    loss.backward()
    allreduce_grads(wm.parameters())
    nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt_wm.step()
    m = w["mask"][:, -1]
    return loss.detach() / args.wm_window, (e_last.detach() * m).sum() / m.sum().clamp(min=1)


def sample_valid_starts(B: int, env_lo: int, env_hi: int):
    starts, hists, n = [], [], 0
    while n < B:
        s, h = buffer.sample_starts(2 * B, Hh, env_lo, env_hi)
        f = s["f"]
        ok = f + HZ <= motion.clip_end[motion.clip_id[f]]
        idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
        starts.append({k: v[idx] for k, v in s.items()}), hists.append({k: v[idx] for k, v in h.items()})
        n += len(idx)
    return starts, hists


def gae_targets(V, rews, dones):
    R = V[-1]
    targets = []
    for t in range(HZ - 1, -1, -1):
        R = rews[t] + args.gamma * (1 - dones[t]) * ((1 - args.lam) * V[t + 1] + args.lam * R)
        targets.append(R)
    return torch.stack(targets[::-1])  # (H, B)


def fit_critic(net, opt, X, targets):
    """X: (H+1, B, D) inputs, targets: (H, B)."""
    net.norm.update(X)
    sync_norms(net.norm)
    Xf, Yf = X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1)
    n = Xf.shape[0]
    mb = n // args.critic_minibatches
    loss = torch.zeros((), device=dev)
    for _ in range(args.critic_epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(args.critic_minibatches):
            idx = perm[i * mb : (i + 1) * mb]
            loss = ((net(Xf[idx]) - Yf[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(net.parameters())
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    return loss.detach()


def discounted_objective(r, d):
    alive = torch.cumprod(1 - d, 0)  # c_{t+1}
    c_t = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), 0)
    disc = args.gamma ** torch.arange(HZ, device=dev, dtype=torch.float32).unsqueeze(-1)
    return (disc * c_t * r).sum(0), alive[-1], (c_t * r).detach().sum(0)


def policy_update():
    sm, hm = sample_valid_starts(BM, 0, N_MO)
    sv, hv = sample_valid_starts(BV, N_MO, args.num_envs) if VEL else ([], [])
    cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}  # noqa: E731
    s, h = cat(sm + sv), cat(hm + hv)
    S = unflatten_state(s["s"])
    xi, f, cmd, cmd_ref = s["xi"], s["f"], s["cmd"], s["cmd_ref"]
    a_prev = s["a_prev"]
    xi_n = normalize_xi(xi)
    with torch.no_grad():
        o_hist = mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"])
    o_hist = mdp.add_noise(o_hist)
    set_requires_grad(wm, False)
    set_requires_grad(critic, False)
    set_requires_grad(critic_v, False)
    S0, f0, cmd0 = S, f, cmd
    Ss, As, Aps, Fs, Cs, align = [], [], [], [], [], 0.0
    for t in range(HZ):
        Aps.append(a_prev), Cs.append(cmd)
        S, a_prev, o_hist, f, cmd, al = imag_step(S, o_hist, f, xi, xi_n, cmd, cmd_ref)
        Ss.append(S), As.append(a_prev), Fs.append(f)
        align = align + al / HZ
    stack = lambda xs, sl: torch.stack([x[sl] for x in xs]).flatten(0, 1)  # noqa: E731  (H*b, ...)
    mo, ve = slice(0, BM), slice(BM, None)
    r, d = reward_term(tuple(stack([St[i] for St in Ss], mo) for i in range(6)), stack(Fs, mo), stack(As, mo), stack(Aps, mo))
    r, d = r.view(HZ, BM), d.view(HZ, BM)
    J, c, r_sum = discounted_objective(r, d)
    J = J + (args.gamma**HZ) * c * critic(_critic_input(tuple(x[mo] for x in S), f[mo], xi_n[mo]))
    loss = -J.mean()
    out = {"imag_rew": (r_sum / HZ / MAX_STEP_REWARD).mean(), "surv": c.mean().detach()}
    if VEL:
        rv, dv = vel_reward_term(tuple(stack([St[i] for St in Ss], ve) for i in range(6)), stack(Cs, ve), stack(As, ve), stack(Aps, ve))
        rv, dv = rv.view(HZ, BV), dv.view(HZ, BV)
        Jv, cv, rv_sum = discounted_objective(rv, dv)
        Jv = Jv + (args.gamma**HZ) * cv * critic_v(_vel_critic_input(tuple(x[ve] for x in S), cmd[ve], xi_n[ve]))
        loss = loss - Jv.mean() + args.align_w * align
        out.update(imag_rew_v=(rv_sum / HZ / MAX_STEP_REWARD_VEL).mean(), surv_v=cv.mean().detach(), align=align.detach())
    opt_pi.zero_grad(set_to_none=True)
    loss.backward()
    allreduce_grads(pi.parameters())
    gn = nn.utils.clip_grad_norm_(pi.parameters(), 1.0)
    nn.utils.clip_grad_value_(pi.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    set_requires_grad(wm, True)
    set_requires_grad(critic, True)
    set_requires_grad(critic_v, True)
    # ---------------- critics: GAE(lambda) targets over the imagined transitions ----------------
    with torch.no_grad():
        traj = lambda i, sl, x0: torch.cat((x0[sl].unsqueeze(0), torch.stack([St[i][sl].detach() for St in Ss])), 0)  # noqa: E731
        S_c = tuple(traj(i, mo, S0[i]).flatten(0, 1) for i in range(6))
        f_c = torch.cat((f0[mo].unsqueeze(0), torch.stack([x[mo] for x in Fs])), 0).flatten()
        xi_c = xi_n[mo].unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
        X = critic_input_c(S_c, f_c, xi_c).view(HZ + 1, BM, -1)
        targets = gae_targets(critic(X), r.detach(), d.detach())
    out["v_loss"] = fit_critic(critic, opt_v, X, targets)
    if VEL:
        with torch.no_grad():
            S_c = tuple(traj(i, ve, S0[i]).flatten(0, 1) for i in range(6))
            cmd_c = torch.cat((torch.stack([x[ve] for x in Cs]), cmd[ve].unsqueeze(0)), 0).flatten(0, 1)
            xi_c = xi_n[ve].unsqueeze(0).expand(HZ + 1, -1, -1).flatten(0, 1)
            X = vel_critic_input_c(S_c, cmd_c, xi_c).view(HZ + 1, BV, -1)
            targets = gae_targets(critic_v(X), rv.detach(), dv.detach())
        out["v_loss_v"] = fit_critic(critic_v, opt_vv, X, targets)
    out.update(pi=loss.detach(), gn=gn.detach())
    return out


# ------------------------------------------------------------------------------------------------
env = TrackingSim(args.num_envs, motion, mdp, dev, gamma=args.gamma, n_vel=N_VEL)
buffer = ReplayBuffer(args.buffer_steps, args.num_envs, dev)
obs_hist = torch.zeros(args.num_envs, Hh, OBS_DIM, device=dev)
is_vel = env.is_vel


@torch.no_grad()
def collect(n_steps: int, stats: dict):
    global obs_hist, sim_steps
    for _ in range(n_steps):
        S = env.state()
        o = mdp.add_noise(mdp.obs(S, env.a_prev, env.xi))
        fresh = env.ep_step == 0
        obs_hist = torch.where(fresh[:, None, None], o.unsqueeze(1).expand_as(obs_hist), torch.cat((obs_hist[:, 1:], o.unsqueeze(1)), 1))
        g = mdp.motion_goal(S[1], env.f, K)
        pi.obs_norm.update(o)
        pi.goal_norm.update(g)
        a = pi(obs_hist[:N_MO], g[:N_MO])
        if VEL:
            a = torch.cat((a, pi.act_vel(obs_hist[N_MO:], mdp.norm_cmd(env.cmd[N_MO:]))), 0)
        a = torch.clamp(a + args.action_noise * torch.randn_like(a), -20, 20)
        out = env.step(a)
        buffer.add(out)
        S0, S1 = unflatten_state(out["s"]), unflatten_state(out["s1"])
        wm.in_norm.update(wm.inputs(S0, out["a"], normalize_xi(out["xi"])))
        wm.out_norm.update(wm.targets(S0, S1))
        sim_steps += args.num_envs * WORLD
        stats["r"].append(out["r"][:N_MO].mean() / MAX_STEP_REWARD)
        if VEL:
            stats["r_v"].append(out["r"][N_MO:].mean() / MAX_STEP_REWARD_VEL)
            stats["cmd_err"].append((mdp.heading_vel(S1)[N_MO:] - out["cmd"][N_MO:]).norm(dim=-1).mean())
        if "ep_len" in out:
            for key, sel in (("", ~out["done_vel"]), ("_v", out["done_vel"])):
                ids = out["done_ids"][sel]
                stats["ep_len" + key].append(out["ep_len"][sel].sum()), stats["ep_ret" + key].append(out["ep_ret_norm"][sel].sum())
                stats["term" + key].append(out["term"][ids].float().sum()), stats["done" + key].append(sel.float().sum())


STAT_KEYS = ["r", "r_v", "cmd_err", "ep_len", "ep_ret", "term", "done", "ep_len_v", "ep_ret_v", "term_v", "done_v"]
stats = {k: [] for k in STAT_KEYS}
if start_it == 0:
    collect(args.warmup_steps, stats)
COLS = ["iteration", "wall_h", "sim_steps", "wm_loss", "wm_err8", "pi_loss", "v_loss", "imag_rew", "pi_gradnorm", "imag_survival",
        "sim_rew", "ep_len", "ep_ret_norm", "term_frac", "v_loss_v", "imag_rew_v", "imag_survival_v", "align",
        "sim_rew_v", "cmd_err", "ep_len_v", "ep_ret_norm_v", "fall_frac_v", "it_per_s", "lr"]  # fmt: skip
if MAIN:
    new_log = not os.path.exists(os.path.join(args.log_dir, "train_log.csv"))
    log_f = open(os.path.join(args.log_dir, "train_log.csv"), "a")
    if new_log:
        log_f.write(",".join(COLS) + "\n")
t_start = time.time() - wall_offset
t_log = time.time()
step_acc = 0.0
agg = {}
stop_file = os.path.join(args.log_dir, "STOP")


def save_ckpt(it: int):
    ck = {**{k: m.state_dict() for k, m in {**MODELS, **OPTS}.items()}, "iteration": it, "wall_time": time.time() - t_start,
          "sim_steps": sim_steps, "args": vars(args), "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
    last = os.path.join(args.log_dir, "last.pt")
    torch.save(ck, last + ".tmp")
    os.replace(last + ".tmp", last)
    small = {k: ck[k] for k in ("pi", "wm", "iteration", "wall_time", "sim_steps", "args", "saved_at")}
    path = os.path.join(args.log_dir, f"ckpt_{it:06d}.pt")
    torch.save(small, path + ".tmp")
    os.replace(path + ".tmp", path)


for it in range(start_it, args.iterations):
    lr = lr_at(it)
    for o in OPTS.values():
        for gparam in o.param_groups:
            gparam["lr"] = lr
    step_acc += 1.0 / args.utd
    n = int(step_acc)
    step_acc -= n
    collect(n, stats)
    sync_norms(pi.obs_norm, pi.goal_norm, wm.in_norm, wm.out_norm)
    for _ in range(args.wm_batches):
        l, e8 = wm_update()
    agg.setdefault("wm", []).append(l), agg.setdefault("wm8", []).append(e8)
    if it >= args.wm_only_iters:
        for k, v in policy_update().items():
            agg.setdefault(k, []).append(v)
    if (it + 1) % args.log_every == 0:
        stop = torch.tensor(float(MAIN and os.path.exists(stop_file)), device=dev)
        if WORLD > 1:
            dist.broadcast(stop, 0)
        if MAIN:
            mean = lambda k: float(sum(float(v) for v in agg.get(k, [])) / max(len(agg.get(k, [])), 1))  # noqa: E731
            ssum = lambda k: float(sum(float(v) for v in stats[k]))  # noqa: E731
            smean = lambda k: float(sum(float(v) for v in stats[k]) / max(len(stats[k]), 1))  # noqa: E731
            ips = args.log_every / (time.time() - t_log)
            t_log = time.time()
            wall_h = (time.time() - t_start) / 3600
            dn, dv = max(ssum("done"), 1), max(ssum("done_v"), 1)
            row = [it + 1, wall_h, sim_steps, mean("wm"), mean("wm8"), mean("pi"), mean("v_loss"), mean("imag_rew"), mean("gn"),
                   mean("surv"), smean("r"), ssum("ep_len") / dn, ssum("ep_ret") / dn, ssum("term") / dn, mean("v_loss_v"),
                   mean("imag_rew_v"), mean("surv_v"), mean("align"), smean("r_v"), smean("cmd_err"), ssum("ep_len_v") / dv,
                   ssum("ep_ret_v") / dv, ssum("term_v") / dv, ips, lr]  # fmt: skip
            log_f.write(",".join(f"{x:.6g}" if isinstance(x, float) else str(x) for x in row) + "\n")
            log_f.flush()
            r = dict(zip(COLS, row))
            print(f"it {it + 1} | {wall_h:.3f} h | wm {r['wm_loss']:.4f} e8 {r['wm_err8']:.4f} | pi {r['pi_loss']:.3f} imag_r {r['imag_rew']:.3f} "
                  f"surv {r['imag_survival']:.3f} | sim_r {r['sim_rew']:.3f} ep_ret {r['ep_ret_norm']:.3f} term {r['term_frac']:.2f} | "
                  f"vel: imag_r {r['imag_rew_v']:.3f} sim_r {r['sim_rew_v']:.3f} cmd_err {r['cmd_err']:.3f} fall {r['fall_frac_v']:.2f} "
                  f"align {r['align']:.4f} | {ips:.2f} it/s", flush=True)  # fmt: skip
        stats = {k: [] for k in STAT_KEYS}
        agg = {}
        if bool(stop):
            if MAIN:
                save_ckpt(it + 1)
                print(f"[stop] STOP file found at it {it + 1}", flush=True)
            break
    if MAIN and ((it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations):
        save_ckpt(it + 1)

if WORLD > 1:
    dist.barrier()
    dist.destroy_process_group()
