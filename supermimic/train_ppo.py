"""PPO terrain teacher. STOP -> atomic last.pt/checkpoint; --resume continues iterations."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage

from supermimic.loco_env import LocoCfg, LocoEnv, STAIRS_OPTIONS, setup_device
from supermimic.elevation_map import ElevationCfg
from supermimic.terrain import TerrainCfg


def runner_config() -> dict:
    model = {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True}
    return {
        "actor": {**model, "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"}},
        "critic": dict(model), "obs_groups": {"actor": ["policy", "height_scan"], "critic": ["critic"]},
        "algorithm": {"class_name": "supermimic.train_ppo:ValidPPO", "value_loss_coef": 1.0,
                      "use_clipped_value_loss": True, "clip_param": .2, "entropy_coef": .01,
                      "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 1e-3,
                      "schedule": "adaptive", "gamma": .99, "lam": .95, "desired_kl": .01,
                      "max_grad_norm": 1.0, "normalize_advantage_per_mini_batch": True,
                      "rnd_cfg": None, "symmetry_cfg": None},
        "num_steps_per_env": 24, "save_interval": 1000, "logger": "tensorboard",
    }


class ValidPPO(PPO):
    """Keep stock PPO losses; omit invalid transitions and cut their GAE traces."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valid = torch.ones_like(self.storage.dones, dtype=torch.bool)
        self.storage.mini_batch_generator = self._valid_batches

    def process_env_step(self, obs, rewards, dones, extras):
        self.valid[self.storage.step, :, 0] = extras["valid"]
        # Stock rsl-rl uses V(s_t) for timeouts. Use the actual pre-reset s_{t+1}.
        rewards = rewards.clone()
        if "terminal_observation" in extras:
            terminal_value = self.critic(extras["terminal_observation"]).detach().squeeze(-1)
            rewards += self.gamma * terminal_value * extras["time_outs"]
        super().process_env_step(obs, rewards, dones, {k: v for k, v in extras.items() if k != "time_outs"})

    def compute_returns(self, obs):
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = torch.zeros_like(last_values)
        for t in reversed(range(st.num_transitions_per_env)):
            next_value = last_values if t + 1 == st.num_transitions_per_env else st.values[t + 1]
            continuation = 1 - st.dones[t].float()
            delta = st.rewards[t] + self.gamma * continuation * next_value - st.values[t]
            advantage = torch.where(self.valid[t], delta + self.gamma * self.lam * continuation * advantage, 0)
            st.returns[t] = advantage + st.values[t]
        st.advantages = st.returns - st.values

    def _valid_batches(self, num_mini_batches, num_epochs):
        st = self.storage
        ids = self.valid.flatten().nonzero().flatten()
        if len(ids) < num_mini_batches * 2:
            raise RuntimeError("Too few valid transitions for PPO")
        obs = st.observations.flatten(0, 1)
        for _ in range(num_epochs):
            shuffled = ids[torch.randperm(len(ids), device=self.device)]
            for idx in shuffled.tensor_split(num_mini_batches):
                yield RolloutStorage.Batch(
                    observations=obs[idx], actions=st.actions.flatten(0, 1)[idx],
                    values=st.values.flatten(0, 1)[idx], advantages=st.advantages.flatten(0, 1)[idx],
                    returns=st.returns.flatten(0, 1)[idx], old_actions_log_prob=st.actions_log_prob.flatten(0, 1)[idx],
                    old_distribution_params=tuple(p.flatten(0, 1)[idx] for p in st.distribution_params))


def cfg_from_dict(values: dict) -> LocoCfg:
    values = dict(values)
    values["terrain"] = TerrainCfg(**values["terrain"])
    if "elevation" in values:
        values["elevation"] = ElevationCfg(**values["elevation"])
    return LocoCfg(**values)


def cfg_to_dict(cfg: LocoCfg) -> dict:
    """Preserve the old serialized config exactly when P2.5 options are defaults."""
    values = asdict(cfg)
    for key in STAIRS_OPTIONS:
        if values[key] == getattr(LocoCfg, key):
            del values[key]
    return values


def add_stairs_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", choices=("stairs_v1",), help="Opt-in stair fine-tuning; applied after resume config")
    descriptions = {
        "foot_clearance_weight": "Swing sole-clearance cost weight (<=0; 0 disables)",
        "foot_clearance_target": "Target sole clearance above terrain, meters",
        "foot_stumble_weight": "Foot horizontal-force-ratio cost weight (<=0; 0 disables)",
        "foot_stumble_ratio": "Stumble when |Fxy| > ratio * |Fz| on either foot",
        "outward_fraction": "Fraction of stairs/slopes commanded forward toward a nearest tile edge",
        "outward_speed": "Outward forward speed in m/s, scaled by command curriculum",
    }
    for key in STAIRS_OPTIONS:
        if key == "contact_vectors":
            parser.add_argument("--contact_vectors", action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument("--" + key, type=float, default=None, help=descriptions[key])


def apply_stairs_args(cfg: LocoCfg, args: argparse.Namespace) -> LocoCfg:
    # G1 MoRE rough config: clearance -0.25 at .10 m, stumble -1 with ratio 3.
    if args.preset == "stairs_v1":
        cfg.foot_clearance_weight, cfg.foot_clearance_target = -.25, .10
        cfg.foot_stumble_weight, cfg.foot_stumble_ratio = -1., 3.
        cfg.outward_fraction, cfg.outward_speed = .5, .6
    for key in STAIRS_OPTIONS:
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    return cfg


class LocoRunner(OnPolicyRunner):
    """Small runner loop for CSV/orchestrator compatibility; rsl-rl owns PPO."""

    def __init__(self, env, cfg, log_dir=None, device="cuda:0"):
        self.train_config = deepcopy(cfg)  # rsl-rl consumes class_name entries
        super().__init__(env, deepcopy(cfg), log_dir=None, device=device)
        self.log_dir = Path(log_dir) if log_dir else None
        self.wall_time = 0.0
        self.logger.writer = None

    def checkpoint(self, iteration: int, args: dict) -> None:
        ck = {**self.alg.save(), "iter": iteration, "iteration": iteration, "infos": {},
              "wall_time": self.wall_time, "args": args, "env_cfg": cfg_to_dict(self.env.cfg),
              "runner_cfg": self.train_config, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "learning_rate": self.alg.learning_rate, "levels": self.env.levels.cpu(),
              "common_step_counter": self.env.common_step_counter,
              "curriculum_up": int(self.env.curriculum_up), "curriculum_down": int(self.env.curriculum_down),
              "total_blown": int(self.env.total_blown), "torch_rng": torch.get_rng_state(),
              "cuda_rng": torch.cuda.get_rng_state(self.device), **self.checkpoint_extra()}
        for name in (f"ckpt_{iteration:06d}.pt", "last.pt"):
            path = self.log_dir / name
            torch.save(ck, str(path) + ".tmp")
            os.replace(str(path) + ".tmp", path)

    def checkpoint_extra(self) -> dict:
        return {}

    def resume(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.alg.load(ck, None, True)
        self.current_learning_iteration, self.wall_time = ck["iter"], ck["wall_time"]
        self.env.common_step_counter = ck.get("common_step_counter", ck["iter"] * self.cfg["num_steps_per_env"])
        self.alg.learning_rate = ck["learning_rate"]
        if len(ck["levels"]) == self.env.num_envs:
            self.env.levels.copy_(ck["levels"])
        else:
            self.env.levels.copy_(ck["levels"][self.env.ids % len(ck["levels"])])
        self.env.curriculum_up.fill_(ck.get("curriculum_up", 0))
        self.env.curriculum_down.fill_(ck.get("curriculum_down", 0))
        self.env.total_blown.fill_(ck.get("total_blown", 0))
        self.env.reset(self.env.ids)
        torch.set_rng_state(ck["torch_rng"].cpu())
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), self.device)

    def train(self, iterations: int, args: dict, save_every: int = 1000, log_every: int = 50) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        obs = self.env.get_observations()
        self.alg.train_mode()
        returns, lengths = deque(maxlen=1000), deque(maxlen=1000)
        accum = {}
        count = 0
        t0, wall0 = time.perf_counter(), self.wall_time
        last_log_time, last_log_it = t0, self.current_learning_iteration
        path = self.log_dir / "train_log.csv"
        exists = path.exists() and path.stat().st_size > 0
        with ExitStack() as files:
            f = files.enter_context(path.open("a", newline=""))
            writer = None
            for it in range(self.current_learning_iteration, iterations):
                if (self.log_dir / "STOP").exists():
                    self.wall_time = wall0 + time.perf_counter() - t0
                    self.checkpoint(it, args)
                    print(f"STOP: saved at iteration {it}", flush=True)
                    break
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        actions = self.alg.act(obs)
                        obs, rewards, dones, extras = self.env.step(actions)
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        for k, v in {"mean_reward": rewards.mean(), **extras["log"]}.items():
                            accum[k] = accum.get(k, 0) + v
                        count += 1
                        returns.extend(extras["episode_returns"].tolist())
                        lengths.extend(extras["episode_lengths"].tolist())
                    self.alg.compute_returns(obs)
                loss = self.alg.update()
                if not all(np.isfinite(v) for v in loss.values()):
                    raise RuntimeError(f"Nonfinite PPO loss: {loss}")
                self.current_learning_iteration = it + 1
                self.wall_time = wall0 + time.perf_counter() - t0
                stop = (self.log_dir / "STOP").exists()
                if (it + 1) % log_every == 0 or it + 1 == iterations or stop:
                    now = time.perf_counter()
                    row = {"iteration": it + 1, "wall_time": self.wall_time, "wall_h": self.wall_time / 3600,
                           **{k: float(v / count) for k, v in accum.items()},
                           "episode_return": float(np.mean(returns)) if returns else 0,
                           "episode_length": float(np.mean(lengths)) if lengths else 0,
                           "it_s": (it + 1 - last_log_it) / (now - last_log_time),
                           "sim_blowups": int(self.env.total_blown), "curriculum_up": int(self.env.curriculum_up),
                           "curriculum_down": int(self.env.curriculum_down), "lr": self.alg.learning_rate,
                           **{f"loss/{k}": v for k, v in loss.items()}}
                    if not all(np.isfinite(v) for v in row.values()):
                        raise RuntimeError("Nonfinite training metrics")
                    if writer is None:
                        fields = list(row)
                        if exists:
                            with path.open(newline="") as previous:
                                reader = csv.DictReader(previous)
                                old_fields = reader.fieldnames
                                if old_fields != fields:
                                    # Fine-tuning can add/remove reward columns. Keep
                                    # old metrics and extend the header atomically.
                                    fields += [k for k in old_fields if k not in row]
                                    temp = path.with_suffix(".csv.tmp")
                                    with temp.open("w", newline="") as updated:
                                        migration = csv.DictWriter(updated, fieldnames=fields)
                                        migration.writeheader()
                                        migration.writerows(reader)
                                    f.close()
                                    os.replace(temp, path)
                                    f = files.enter_context(path.open("a", newline=""))
                        writer = csv.DictWriter(f, fieldnames=fields)
                        if not exists:
                            writer.writeheader()
                    writer.writerow(row)
                    f.flush()
                    print(f"it {it+1} | reward {row['mean_reward']:.4f} return {row['episode_return']:.2f} "
                          f"length {row['episode_length']:.1f} | level {row['terrain_level']:.3f} "
                          f"up {row['curriculum_up']} | err {row['tracking_lin']:.3f} | "
                          f"blowups {row['sim_blowups']} | {row['it_s']:.2f} it/s", flush=True)
                    accum, count, last_log_time, last_log_it = {}, 0, now, it + 1
                if (it + 1) % save_every == 0 or it + 1 == iterations or stop:
                    self.checkpoint(it + 1, args)
                if stop:
                    print(f"STOP: saved at iteration {it+1}", flush=True)
                    break


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", default="logs/loco_teacher")
    p.add_argument("--collision", choices=("mesh", "proxy"), default="proxy")
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--terrain", choices=("full", "flat"), default="full")
    p.add_argument("--max_init_level", type=int, default=0)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    add_stairs_args(p)
    args = p.parse_args()
    setup_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = LocoCfg(seed=args.seed, collision=args.collision, max_init_level=args.max_init_level)
    train_cfg = runner_config()
    if args.terrain == "flat":
        cfg.terrain.proportions = (1., 0., 0., 0., 0., 0., 0.)
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg, train_cfg = cfg_from_dict(ck["env_cfg"]), ck["runner_cfg"]
    apply_stairs_args(cfg, args)
    log = Path(args.log_dir)
    log.mkdir(parents=True, exist_ok=True)
    (log / "args.json").write_text(json.dumps({**vars(args), "env_cfg": cfg_to_dict(cfg), "runner_cfg": train_cfg}, indent=2))
    env = LocoEnv(args.num_envs, cfg, args.device)
    runner = LocoRunner(env, train_cfg, args.log_dir, args.device)
    if args.resume:
        runner.resume(args.resume)
    runner.train(args.iterations, vars(args), args.save_every, args.log_every)


if __name__ == "__main__":
    main()
