"""Deterministic terrain-by-level evaluation; --watch scores newly saved checkpoints."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supermimic.loco_env import LocoEnv, setup_device
from supermimic.terrain import TILE_TYPES
from supermimic.train_ppo import LocoRunner, cfg_from_dict
from supermimic.student import is_student, load_student


DIRECTION = "legacy"
TAG = {"legacy": "", "forward": "_fwd"}
YAW_DEG = 5.0  # start headings spread uniformly over +/- YAW_DEG around the tile's x axis


def tag() -> str:
    return TAG[DIRECTION] + (f"_yaw{YAW_DEG:g}" if YAW_DEG != 5.0 else "")


@torch.no_grad()
def evaluate(checkpoint: str | Path, trials: int = 8, device: str = "cuda:0", map_noise: bool = True) -> dict:
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    torch.manual_seed(0)
    cfg = cfg_from_dict(deepcopy(ck["env_cfg"]))
    cfg.curriculum = cfg.randomization = cfg.observation_noise = False
    student = is_student(ck)
    if student:
        cfg.elevation.noise = map_noise
    cfg.episode_seconds = 20
    # Always evaluate all seven P1 types, including checkpoints trained on flat.
    cfg.terrain.proportions = (0.10, 0.20, 0.15, 0.15, 0.15, 0.15, 0.10)
    cfg.terrain.num_cols = max(20, cfg.terrain.num_cols)
    rows = cfg.terrain.num_rows
    env = LocoEnv(rows * len(TILE_TYPES) * trials, cfg, device, eval_mode=True)
    cases = torch.cartesian_prod(torch.arange(rows, device=device), torch.arange(len(TILE_TYPES), device=device))
    levels, kinds = cases.repeat_interleave(trials, dim=0).T
    columns = torch.stack([(env.tile_types[0] == k).nonzero()[0, 0] for k in range(len(TILE_TYPES))])
    env.levels.copy_(levels)
    env.columns.copy_(columns[kinds])
    env.kinds.copy_(kinds)
    env.reset(env.ids)
    # Reproducible +/- 5 degree headings provide distinct trials at the exact center.
    half = YAW_DEG * torch.pi / 180
    yaw = torch.linspace(-half, half, trials, device=device).repeat(len(cases)) if trials > 1 else torch.zeros(env.num_envs, device=device)
    env.qpos[:, 3], env.qpos[:, 6] = (yaw / 2).cos(), (yaw / 2).sin()
    env.cmd.zero_()
    backwards = (kinds == TILE_TYPES.index("slope_down")) | (kinds == TILE_TYPES.index("stairs_down"))
    # legacy: walk backwards out of the "down" pyramids (i.e. climb them backwards); forward: always walk forward,
    # so stairs_up/slope_up tiles test descending and stairs_down/slope_down tiles test ascending
    env.cmd[:, 0] = .6 if DIRECTION == "forward" else torch.where(backwards, -.6, .6)
    if student:
        policy = load_student(ck, device)
    else:
        runner = LocoRunner(env, ck["runner_cfg"], device=device)
        runner.alg.load(ck, {"actor": True, "critic": True}, True)
        policy = runner.get_inference_policy(device)
    start = env.qpos[:, :2].clone()
    fell = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    lin_sum, yaw_sum, samples = [torch.zeros(env.num_envs, device=device) for _ in range(3)]
    max_distance = torch.zeros(env.num_envs, device=device)
    obs = env.get_observations()
    for _ in range(400):
        alive = ~fell
        obs, _, _, extras = env.step(policy(obs))
        fell |= extras["falls"]
        lin_sum += torch.where(alive, extras["tracking_lin"], 0)
        yaw_sum += torch.where(alive, extras["tracking_yaw"], 0)
        samples += alive
        distance = (env.qpos[:, :2] - start).norm(dim=-1)
        max_distance = torch.maximum(max_distance, torch.where(alive, distance, 0))
        # Reset failed worlds for numerical hygiene; exclude every subsequent step.
        env.reset(extras["falls"].nonzero().flatten())
    success = (max_distance > 3) & ~fell
    records = []
    for level in range(rows):
        for k, name in enumerate(TILE_TYPES):
            mask = (levels == level) & (kinds == k)
            records.append({"iteration": ck["iter"], "terrain": name, "level": level, "trials": trials,
                            **({"map_noise": "on" if map_noise else "off"} if student else {}),
                            "success": float(success[mask].float().mean()), "falls": int(fell[mask].sum()),
                            "tracking_error": float((lin_sum[mask] / samples[mask].clamp(min=1)).mean()),
                            "yaw_error": float((yaw_sum[mask] / samples[mask].clamp(min=1)).mean()),
                            "distance": float(max_distance[mask].mean())})
    return {"checkpoint": str(checkpoint), "iteration": ck["iter"], "seconds": 8, "speed": .6,
            **({"map_noise": "on" if map_noise else "off", "policy_type": "student"} if student else {}),
            "trials": trials, "seed": 0, "success": float(success.float().mean()), "falls": int(fell.sum()),
            "num_envs": env.num_envs, "blowups": int(env.total_blown), "results": records}


def write_result(checkpoint: Path, result: dict) -> None:
    suffix = "_eval" + tag() + ("_noise_" + result["map_noise"] if "map_noise" in result else "") + ".json"
    destination = checkpoint.with_name(checkpoint.stem + suffix)
    temp = destination.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(destination)
    csv_path = checkpoint.parent / f"eval_loco{tag()}.csv"
    # Re-evaluation replaces this iteration instead of accumulating duplicate rows.
    previous = []
    if csv_path.exists():
        with csv_path.open() as f:
            previous = [r for r in csv.DictReader(f) if int(r["iteration"]) != result["iteration"]
                        or r.get("map_noise") != result.get("map_noise")]
    temp_csv = csv_path.with_suffix(".tmp")
    with temp_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["results"][0]))
        writer.writeheader()
        writer.writerows(previous + result["results"])
    temp_csv.replace(csv_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", type=Path)
    p.add_argument("--ckpt", type=Path)
    p.add_argument("--watch", action="store_true")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--poll_seconds", type=float, default=30)
    p.add_argument("--map_noise", choices=("on", "off", "both"), default="on")
    p.add_argument("--direction", choices=("legacy", "forward"), default="legacy")
    p.add_argument("--yaw_deg", type=float, default=5.0, help="start heading spread (+/- degrees)")
    args = p.parse_args()
    global DIRECTION, YAW_DEG
    DIRECTION, YAW_DEG = args.direction, args.yaw_deg
    if args.trials < 1 or not (args.run_dir or args.ckpt):
        p.error("Provide --run_dir or --ckpt, and positive --trials")
    setup_device(args.device)
    while True:
        checkpoints = [args.ckpt] if args.ckpt else sorted(args.run_dir.glob("ckpt_*.pt"))
        todo = []
        for checkpoint in checkpoints:
            ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
            modes = ([True, False] if args.map_noise == "both" else [args.map_noise == "on"]) if is_student(ck) else [None]
            for mode in modes:
                suffix = "_eval" + tag() + ("_noise_" + ("on" if mode else "off") if mode is not None else "") + ".json"
                if args.ckpt or not checkpoint.with_name(checkpoint.stem + suffix).exists():
                    todo.append((checkpoint, mode))
        if args.watch and todo:
            todo = [(c, mode) for c, mode in todo if c == todo[-1][0]]
        for checkpoint, mode in todo:
            start = time.perf_counter()
            result = evaluate(checkpoint, args.trials, args.device, map_noise=mode is not False)
            write_result(checkpoint, result)
            print(f"{checkpoint}: success {result['success']:.3f}, falls {result['falls']}/{result['num_envs']} "
                  f"({time.perf_counter()-start:.1f}s)", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        if not args.watch or args.ckpt:
            break
        time.sleep(min(args.poll_seconds, 60))


if __name__ == "__main__":
    main()
