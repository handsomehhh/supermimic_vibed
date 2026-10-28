"""DAgger distillation and optional asymmetric PPO for the elevation-map student."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from rsl_rl.runners import DistillationRunner

from supermimic.loco_env import LocoEnv, setup_device
from supermimic.train_ppo import LocoRunner, cfg_from_dict, runner_config
from supermimic.student import is_student
from supermimic.terrain import TILE_TYPES


def student_config():
    return {"class_name": "supermimic.student:StudentModel", "hidden_dims": [256, 128],
            "activation": "elu", "obs_normalization": True, "map_latent": 64}


def distillation_config(teacher):
    return {"policy_type": "student", "stage": "distill", "student": student_config(),
            "teacher": deepcopy(teacher["runner_cfg"]["actor"]),
            "obs_groups": {"student": ["policy_history", "elevation_map"],
                           "teacher": deepcopy(teacher["runner_cfg"]["obs_groups"]["actor"])},
            "algorithm": {"class_name": "Distillation", "num_learning_epochs": 2,
                          "gradient_length": 1, "learning_rate": 3e-4, "max_grad_norm": 1., "loss_type": "mse"},
            "num_steps_per_env": 24, "save_interval": 1000, "logger": "tensorboard"}


def finetune_config(student):
    cfg = runner_config()
    cfg.update(policy_type="student", stage="ppo")
    cfg["actor"] = deepcopy(student["runner_cfg"].get("student", student["runner_cfg"].get("actor")))
    cfg["actor"]["distribution_cfg"] = {"class_name": "GaussianDistribution", "init_std": .1, "std_type": "scalar"}
    cfg["critic"] = deepcopy(student.get("teacher_critic_cfg", student["runner_cfg"].get("critic", cfg["critic"])))
    cfg["obs_groups"]["actor"] = ["policy_history", "elevation_map"]
    cfg["algorithm"].update(learning_rate=1e-4, entropy_coef=.001)
    return cfg


class StudentRunner(LocoRunner, DistillationRunner):
    """DistillationRunner algorithm with the P2 CSV/checkpoint/STOP training loop."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_critic = {}

    def checkpoint_extra(self):
        return self.teacher_critic

    def train(self, *args, **kwargs):
        if not self.alg.teacher_loaded:
            raise ValueError("Load a P2 teacher before distillation")
        return super().train(*args, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("distill", "ppo"), default="distill")
    p.add_argument("--teacher", help="P2 teacher checkpoint (distillation only)")
    p.add_argument("--student", help="Distilled student checkpoint to initialize PPO")
    p.add_argument("--resume", help="Resume the selected stage; --iterations is its total target")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--history_length", type=int, default=5)
    p.add_argument("--map_config", type=Path, help="JSON overrides for ElevationCfg (new distillation runs)")
    p.add_argument("--map_noise", choices=("on", "off"), default="on")
    p.add_argument("--map_encoder", choices=("mlp", "cnn"), default="mlp", help="Map encoder (new distillation runs)")
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128], help="Policy MLP widths (new distillation runs)")
    p.add_argument("--map_latent", type=int, default=64, help="Map latent size (new distillation runs)")
    p.add_argument("--terrain_proportions", type=float, nargs=len(TILE_TYPES),
                   help="Terrain weights, including on resume: " + ", ".join(TILE_TYPES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log_dir", required=True)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    args = p.parse_args()
    source = args.resume or (args.teacher if args.stage == "distill" else args.student)
    if not source or min(args.num_envs, args.history_length, args.save_every, args.log_every) < 1 or args.iterations < 0:
        p.error("Provide --teacher for distill, --student for ppo, or --resume; counts must be positive")
    if min(args.hidden_dims) < 1 or args.map_latent < 1:
        p.error("hidden_dims and map_latent must be positive")
    if args.terrain_proportions is not None:
        weights = np.asarray(args.terrain_proportions)
        if not np.isfinite(weights).all() or (weights < 0).any() or not 0 < weights.sum() < np.inf:
            p.error("terrain_proportions must be finite nonnegative weights with positive sum")
    ck = torch.load(source, map_location="cpu", weights_only=False)
    cfg = cfg_from_dict(ck["env_cfg"])
    if args.resume:
        if not is_student(ck) or ck["runner_cfg"]["stage"] != args.stage:
            p.error("Resume checkpoint must be a student from the selected stage")
        train_cfg = ck["runner_cfg"]
    elif args.stage == "distill":
        if is_student(ck):
            p.error("--teacher must be a P2 teacher checkpoint")
        cfg.elevation.enabled, cfg.elevation.noise = True, args.map_noise == "on"
        cfg.history_length, cfg.seed = args.history_length, args.seed
        if args.map_config:
            from supermimic.elevation_map import ElevationCfg
            cfg.elevation = ElevationCfg(**{**asdict(cfg.elevation), **json.loads(args.map_config.read_text())})
            cfg.elevation.enabled = True
        train_cfg = distillation_config(ck)
        train_cfg["student"].update(map_encoder=args.map_encoder, hidden_dims=args.hidden_dims, map_latent=args.map_latent)
        if args.map_encoder == "cnn":
            train_cfg["student"]["map_shape"] = list(cfg.elevation.shape)
    else:
        if not is_student(ck):
            p.error("--student must be a P3 student checkpoint")
        cfg.elevation.noise = args.map_noise == "on"
        train_cfg = finetune_config(ck)
    if args.terrain_proportions is not None:
        cfg.terrain.proportions = tuple(args.terrain_proportions)
    setup_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    log = Path(args.log_dir)
    log.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(log.glob("ckpt_*.pt")):
        p.error("Use a new log directory for a new stage, or --resume")
    (log / "args.json").write_text(json.dumps({**vars(args), "map_config": str(args.map_config) if args.map_config else None,
                                             "env_cfg": asdict(cfg), "runner_cfg": train_cfg}, indent=2))
    env = LocoEnv(args.num_envs, cfg, args.device)
    runner = (StudentRunner if args.stage == "distill" else LocoRunner)(env, train_cfg, log, args.device)
    if args.stage == "distill":
        runner.teacher_critic = ({k: ck[k] for k in ("teacher_critic_state_dict", "teacher_critic_cfg")} if args.resume else
                                {"teacher_critic_state_dict": ck["critic_state_dict"],
                                 "teacher_critic_cfg": deepcopy(ck["runner_cfg"]["critic"])})
    if args.resume:
        runner.resume(args.resume)
    elif args.stage == "distill":
        runner.alg.load(ck, {"teacher": True}, True)
        runner.alg.teacher.requires_grad_(False)
    else:
        state = ck.get("student_state_dict", ck.get("actor_state_dict"))
        missing, unexpected = runner.alg.actor.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("distribution.") for k in missing):
            raise ValueError(f"Incompatible student weights: {missing}, {unexpected}")
        critic = ck.get("teacher_critic_state_dict", ck.get("critic_state_dict"))
        runner.alg.critic.load_state_dict(critic)
        # Retain command curriculum progress; stage iteration/log clocks restart.
        env.common_step_counter = ck.get("common_step_counter", 0)
    if not args.resume:
        env.common_step_counter = ck.get("common_step_counter", 0)
        env.levels.copy_(ck["levels"][env.ids.cpu() % len(ck["levels"])].to(env.device))
        env.reset(env.ids)
        runner.checkpoint(0, vars(args) | {"map_config": str(args.map_config) if args.map_config else None})
    runner.train(args.iterations, vars(args), args.save_every, args.log_every)


if __name__ == "__main__":
    main()
