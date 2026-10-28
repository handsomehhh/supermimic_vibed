"""Record forward ascent on stairs_down, including forces and pre-fall sole paths.

Use --ckpt for a walking teacher/student, or --synthetic for a default-pose robot
given an initial forward velocity near the first riser (a recorder smoke test).
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from supermimic.loco_env import LocoCfg, LocoEnv, setup_device
from supermimic.student import is_student, load_student
from supermimic.terrain import TILE_TYPES
from supermimic.train_ppo import LocoRunner, cfg_from_dict, cfg_to_dict


def summarize(trace: dict, levels, trials: int, dt: float, ratio: float) -> list[dict]:
    records = []
    for i, level in enumerate(levels):
        valid = trace["alive"][:, i] & trace["valid"][:, i]
        force = trace["foot_force"][:, i]
        horizontal = np.linalg.norm(force[..., :2], axis=-1)
        stumble = (horizontal > ratio * np.abs(force[..., 2])).any(-1) & valid
        shin = (trace["shin_normal"][:, i] > 10).any(-1) & valid
        fallen = bool(trace["falls"][:, i].any())
        invalid = bool((trace["alive"][:, i] & ~trace["valid"][:, i]).any())
        ids = np.flatnonzero(valid)
        last = int(ids[-1]) if len(ids) else 0
        window = np.arange(len(valid)) >= max(0, last - round(.5 / dt))
        distance = np.linalg.norm(trace["qpos"][:, i, :2] - trace["start_xy"][i], axis=-1)
        maximum = float(distance[valid].max()) if valid.any() else 0.
        recent = max(0, last - round(1. / dt))
        progress = float(np.linalg.norm(trace["qpos"][last, i, :2] - trace["qpos"][recent, i, :2]))
        incomplete = "stall" if progress < .05 else "incomplete"
        outcome = "invalid" if invalid else ("fall" if fallen else ("success" if maximum > 3 else incomplete))
        triggers = []
        if invalid:
            triggers.append("numerical_guard")
        if fallen and valid.any():
            if trace["base_height"][last, i] < .35:
                triggers.append("base_height_below_0.35_m")
            if trace["tilt_deg"][last, i] > 70:
                triggers.append("tilt_above_70_degrees")
        suspect = []
        if (stumble & window).any():
            suspect.append("foot horizontal-force ratio exceeded before end; possible riser catch")
        if (shin & window).any():
            suspect.append("knee/shin contact before end")
        records.append({"env": i, "level": int(level), "trial": i % trials, "outcome": outcome,
                        "end_time": float(trace["time"][last]), "max_distance": maximum,
                        "end_xy_from_center": (trace["qpos"][last, i, :2] - trace["origins"][i, :2]).tolist(),
                        "end_base_z": float(trace["qpos"][last, i, 2]),
                        "end_base_height_m": float(trace["base_height"][last, i]),
                        "end_tilt_deg": float(trace["tilt_deg"][last, i]), "termination_triggers": triggers,
                        "last_second_progress_m": progress,
                        "stumble_steps": int(stumble.sum()), "shin_contact_steps": int(shin.sum()),
                        "peak_foot_horizontal_N": float(horizontal[valid].max()) if valid.any() else None,
                        "peak_shin_normal_N": float(trace["shin_normal"][valid, i].max()) if valid.any() else None,
                        "hypotheses": suspect or ["no force-based cause identified; inspect trajectory"]})
    return records


def plot_cases(trace: dict, records: list[dict], out: Path, limit: int) -> list[str]:
    if limit <= 0:
        return []
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Show failures from each requested level before adding more trials of one
    # level; retain successful cases as useful comparisons when failures run out.
    groups = [sorted([r for r in records if r["level"] == level], key=lambda r: r["outcome"] == "success")
              for level in sorted({r["level"] for r in records})]
    selected = [group[j] for j in range(max(map(len, groups))) for group in groups if j < len(group)][:limit]
    names = []
    for record in selected:
        i = record["env"]
        valid = trace["alive"][:, i] & trace["valid"][:, i]
        t = trace["time"][valid]
        origin = trace["origins"][i]
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
        root_x = trace["qpos"][valid, i, 0] - origin[0]
        hi = max(1.3, float(root_x.max()) + .4) if valid.any() else 1.3
        profile_mask = trace["profile_x"] <= hi
        axes[0].plot(trace["profile_x"][profile_mask], trace["profile_z"][i, profile_mask] - origin[2],
                     "k", label="terrain at tile-center y")
        axes[0].plot(root_x, trace["qpos"][valid, i, 2] - origin[2], "g--", label="base path")
        for foot, color, name in ((0, "tab:blue", "left"), (1, "tab:orange", "right")):
            points = trace["sole_xyz"][valid, i, foot]
            ground = trace["sole_terrain_z"][valid, i, foot]
            axes[0].scatter(points[..., 0] - origin[0], points[..., 2] - origin[2], s=3, c=color, label=f"{name} sole pads")
            axes[0].scatter(points[..., 0] - origin[0], ground - origin[2], s=2, c="gray", alpha=.3)
            forces = trace["foot_force"][valid, i, foot]
            axes[1].plot(t, np.linalg.norm(forces[:, :2], axis=-1), color=color, label=f"{name} foot |Fxy|")
            axes[1].plot(t, forces[:, 2], color=color, linestyle=":", label=f"{name} foot Fz")
            axes[1].plot(t, trace["shin_normal"][valid, i, foot], color=color, linestyle="--", label=f"{name} shin normal")
            axes[2].plot(t, trace["clearance"][valid, i, foot], color=color, label=f"{name} min sole clearance")
        axes[0].set(xlabel="World x from tile center (m)", ylabel="Height above center (m)")
        axes[1].set(xlabel="Time (s)", ylabel="Contact force (N; 20 ms average)")
        axes[2].set(xlabel="Time (s)", ylabel="Sole clearance (m)")
        axes[2].axhline(.1, color="gray", linestyle=":", label="0.10 m reference")
        for ax in axes:
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=.2)
        fig.suptitle(f"Level {record['level']}, trial {record['trial']}: {record['outcome']}\n"
                     + textwrap.fill("; ".join(record["hypotheses"]), width=95), fontsize=11)
        name = f"level{record['level']}_trial{record['trial']}.png"
        fig.savefig(out / name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        names.append(name)
    return names


@torch.no_grad()
def diagnose(args) -> dict:
    torch.manual_seed(0)
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False) if args.ckpt else None
    cfg = cfg_from_dict(deepcopy(ck["env_cfg"])) if ck else LocoCfg()
    if min(args.levels) < 0 or max(args.levels) >= cfg.terrain.num_rows:
        raise ValueError("Requested level is outside the checkpoint terrain rows")
    cfg.curriculum = cfg.randomization = cfg.observation_noise = False
    cfg.contact_vectors = True
    cfg.episode_seconds = max(20., args.seconds + 1.)
    cfg.terrain.proportions = (.10, .20, .15, .15, .15, .15, .10)
    cfg.terrain.num_cols = max(20, cfg.terrain.num_cols)
    student = ck is not None and is_student(ck)
    if student:
        cfg.elevation.noise = args.map_noise == "on"
    levels = torch.tensor(args.levels, device=args.device).repeat_interleave(args.trials)
    env = LocoEnv(len(levels), cfg, args.device, eval_mode=True)
    env.levels.copy_(levels)
    kind = TILE_TYPES.index("stairs_down")
    column = (env.tile_types[0] == kind).nonzero()[0, 0]
    env.columns.fill_(int(column))
    env.kinds.fill_(kind)
    env.reset(env.ids)
    yaw = (torch.linspace(-np.deg2rad(5), np.deg2rad(5), args.trials, device=args.device)
           if args.trials > 1 else torch.zeros(1, device=args.device)).repeat(len(args.levels))
    env.qpos[:, 3], env.qpos[:, 6] = (yaw / 2).cos(), (yaw / 2).sin()
    env.cmd[:, 0] = args.speed
    if args.synthetic:
        env.qpos[:, 0] += .60
        env.qvel[:, 0] = .8
    with wp.ScopedDevice(args.device):
        mjw.kinematics(env.model, env.data)
    if student:
        policy = load_student(ck, args.device)
    elif ck:
        runner = LocoRunner(env, ck["runner_cfg"], device=args.device)
        runner.alg.load(ck, {"actor": True, "critic": True}, True)
        policy = runner.get_inference_policy(args.device)
    else:
        policy = lambda obs: torch.zeros(env.num_envs, 29, device=args.device)
    shin_ids = [env.mjm.body(side + "_knee_link").id for side in ("left", "right")]
    start = env.qpos[:, :2].clone()
    origins = env.origins[env.levels, env.columns].clone()
    profile_x = torch.linspace(-.3, 4.1, 441, device=args.device)
    xy = origins[:, None, :2].expand(-1, len(profile_x), -1).clone()
    xy[..., 0] += profile_x
    static = {"start_xy": start, "origins": origins, "profile_x": profile_x,
              "profile_z": env.sampler.sample(xy), "levels": levels}
    collected = {}
    failed = torch.zeros(env.num_envs, device=args.device, dtype=torch.bool)
    obs = env.get_observations()
    steps = round(args.seconds / env.dt)
    for _ in range(steps):
        alive = ~failed
        action = policy(obs)
        obs, _, _, extras = env.step(action)
        points, ground, clearance = env.foot_clearance()
        _, _, gravity, height = env._state()
        state = {"qpos": env.qpos, "qvel": env.qvel, "actions": action, "sole_xyz": points,
                 "sole_terrain_z": ground, "clearance": clearance,
                 "base_height": height, "tilt_deg": torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1, 1))),
                 "foot_force": env.body_force_vector[:, env.feet], "foot_normal": env.body_force[:, env.feet],
                 "shin_force": env.body_force_vector[:, shin_ids], "shin_normal": env.body_force[:, shin_ids],
                 "alive": alive, "valid": extras["valid"], "falls": extras["falls"] & alive}
        for key, value in state.items():
            collected.setdefault(key, []).append(value.cpu().numpy().copy())
        failed |= extras["falls"]
        # Preserve the first fall, then ignore reset states. Invalid steps have
        # already rolled back in LocoEnv; the valid mask explicitly excludes them.
        env.reset(extras["falls"].nonzero().flatten())
        obs = env.get_observations()
    trace = {key: np.stack(value) for key, value in collected.items()}
    trace.update({key: value.cpu().numpy() for key, value in static.items()})
    trace["time"] = np.arange(1, steps + 1) * env.dt
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "trajectory.npz", **trace)
    records = summarize(trace, levels.cpu().tolist(), args.trials, env.dt, cfg.foot_stumble_ratio)
    result = {"checkpoint": str(args.ckpt) if ck else None, "synthetic": args.synthetic,
              "synthetic_description": "default joint pose, x offset .60 m, initial vx .8 m/s" if args.synthetic else None,
              "direction": "forward", "terrain": "stairs_down", "speed": args.speed,
              "seconds": steps * env.dt, "env_cfg": cfg_to_dict(cfg), "seed": 0,
              "force_convention": "world-frame net vectors and summed normal magnitudes, averaged over four substeps",
              "trace_masks": "alive includes the first fall; valid excludes numerical rollbacks; ignore later samples",
              "hypothesis_caveat": "Force flags are correlations, not a causal diagnosis. Inspect the saved pre-fall traces.",
              "blowups": int(env.total_blown), "results": records,
              "plots": plot_cases(trace, records, args.out, args.plots)}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--ckpt", type=Path)
    source.add_argument("--synthetic", action="store_true")
    p.add_argument("--levels", nargs="+", type=int, default=[7, 8, 9])
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--seconds", type=float, default=8.)
    p.add_argument("--speed", type=float, default=.6)
    p.add_argument("--plots", type=int, default=6)
    p.add_argument("--map_noise", choices=("on", "off"), default="on")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.trials < 1 or not np.isfinite([args.seconds, args.speed]).all() or args.seconds < .02 or args.speed <= 0:
        p.error("Require positive trials/speed and seconds >= .02")
    setup_device(args.device)
    result = diagnose(args)
    for row in result["results"]:
        print(f"level {row['level']} trial {row['trial']}: {row['outcome']}, "
              f"distance {row['max_distance']:.2f} m, stumble {row['stumble_steps']}, shin {row['shin_contact_steps']}")
    print(f"Saved {args.out / 'summary.json'} and trajectory.npz", flush=True)


if __name__ == "__main__":
    main()
