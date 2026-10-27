"""P3 map semantics/visualization and CPU monitor fixtures; writes local evidence."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from supermimic.elevation_map import ElevationCfg, ElevationMap


def maps(out, device):
    from supermimic.loco_env import LocoCfg, LocoEnv, setup_device
    from supermimic.rotations import quat_to_mat
    from supermimic.terrain import TILE_TYPES
    setup_device(device)
    torch.manual_seed(42)
    cfg = LocoCfg(randomization=False, observation_noise=False, elevation=ElevationCfg(enabled=True))
    env = LocoEnv(128, cfg, device, eval_mode=True)
    column = (env.tile_types[0] == TILE_TYPES.index("stairs_up")).nonzero()[0, 0]
    env.columns[:] = column
    env.levels[:] = 9
    env.reset(env.ids)
    env.qpos[:, 0] -= 2.6  # Face +x toward the rising side of the pyramid stairs.
    env.qpos[:, 2] = env.sampler.sample(env.qpos[:, :2]) + env.stand_height
    obs = env.get_observations()
    m = env.elevation
    cells = m.cells
    exact = -env.sampler.scan(env.qpos[:, :3], quat_to_mat(env.qpos[:, 3:7]), m.grid)
    assert (exact[:, -15:].mean() - exact[:, :15].mean()) > .5
    valid = obs["elevation_map"][:, cells:].bool()
    fractions = {name: float(getattr(m, name + "_mask").float().mean()) for name in ("dropout", "blind", "shadow")}
    parameters = {"cell_sigma_range_m": [float(m.sigma.min()), float(m.sigma.max())],
                  "xy_error_max_abs_m": float(m.shift.abs().max()),
                  "yaw_error_max_abs_degrees": float(m.angle.abs().max() * 180 / torch.pi),
                  "latency_steps_range": [int(m.latency.min()), int(m.latency.max())],
                  "refresh_steps_range": [int(m.period.min()), int(m.period.max())]}
    assert (obs["elevation_map"][:, :cells][~valid] == 0).all()
    again = env.get_observations()
    for key in ("policy_history", "policy", "elevation_map"):
        torch.testing.assert_close(obs[key], again[key], rtol=0, atol=0)
    # Returned observations own their memory, including across partial resets.
    original = obs["policy_history"].clone()
    old_map = obs["elevation_map"].clone()
    env.reset(env.ids[:1])
    reset_obs = env.get_observations()
    assert (reset_obs["policy_history"][0].reshape(5, 96) == reset_obs["policy"][0]).all()
    torch.testing.assert_close(obs["policy_history"], original, rtol=0, atol=0)
    torch.testing.assert_close(reset_obs["elevation_map"][1:], old_map[1:], rtol=0, atol=0)
    # Actual physics step advances history once; repeated reads do not advance it.
    env.reset(env.ids)
    before = env.get_observations()["policy_history"].reshape(128, 5, 96).clone()
    stepped, _, _, _ = env.step(torch.zeros(128, 29, device=device))
    after = stepped["policy_history"].reshape(128, 5, 96)
    torch.testing.assert_close(after[:, :-1], before[:, 1:], rtol=0, atol=0)
    env.max_episode_length = 2
    _, _, _, extra = env.step(torch.zeros(128, 29, device=device))
    assert "terminal_observation" in extra
    terminal = extra["terminal_observation"]["policy_history"].reshape(128, 5, 96)
    torch.testing.assert_close(terminal[:, :-1], after[:, 1:], rtol=0, atol=0)

    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    images = (exact[1], old_map[1, :cells], old_map[1, cells:])
    for ax, values, title in zip(axes, images, ("Exact height on map grid", "Noisy onboard height", "Validity mask")):
        kwargs = {"vmin": float(exact[1].min()), "vmax": float(exact[1].max())} if title != "Validity mask" else {"vmin": 0, "vmax": 1}
        im = ax.imshow(values.reshape(*cfg.elevation.shape).T.cpu(), origin="lower", extent=(-.5, 1.5, -.6, .6), **kwargs)
        ax.set(title=title, xlabel="Forward x (m)", ylabel="Left y (m)")
        fig.colorbar(im, ax=ax, label="terrain_z - base_z (m)" if title != "Validity mask" else "valid")
    fig.savefig(out / "stairs_map.png", dpi=160)
    plt.close(fig)
    invalid_fraction = float((~valid).float().mean())

    class Flat:
        def sample(self, xy):
            return torch.zeros(xy.shape[:-1], device=xy.device)
    c = ElevationCfg(enabled=True, cell_noise=0, drift=0, xy_error=0, yaw_error_degrees=0,
                     dropout=(0, 0), blind_radii=(.001, .001), refresh_seconds=(.1, .1))
    delayed = ElevationMap(c, 2, device, .02, Flat())
    ids = torch.arange(2, device=device)
    delayed.reset(ids)
    delayed.latency[:] = torch.tensor([0, 3], device=device)
    delayed.period[:] = torch.tensor([4, 5], device=device)
    position = torch.zeros(2, 3, device=device)
    yaw = torch.zeros(2, device=device)
    timeline = []
    for tick in range(16):
        position[:, 2] = 1 + .1 * tick
        value = delayed.observe(position, yaw, tick)
        acquisition = torch.tensor([tick // 4 * 4, max(0, tick // 5 * 5 - 3)], device=device)
        expected = -1 - acquisition * .1
        torch.testing.assert_close(value[:, :delayed.cells], expected[:, None].expand(-1, delayed.cells))
        assert (delayed.acquisition_step == acquisition).all()
        timeline.append({"tick": tick, "refresh": delayed.refresh_step.tolist(), "acquisition": delayed.acquisition_step.tolist(),
                         "height": value[:, 0].tolist()})
    delayed.reset(ids[:1])
    position[0, 2] = 9
    value = delayed.observe(position, yaw, 16)
    torch.testing.assert_close(value[0, :delayed.cells], torch.full((delayed.cells,), -9., device=device))
    delayed.cfg.noise = False
    for tick in range(17, 20):
        position[:, 2] += .1
        value = delayed.observe(position, yaw, tick)
        torch.testing.assert_close(value[:, :delayed.cells], -position[:, 2:3].expand(-1, delayed.cells))
        assert (value[:, delayed.cells:] == 1).all()

    class Wall:
        def sample(self, xy):
            return ((xy[..., 0] > .35) & (xy[..., 0] < .55)).float() * 1.3
    shadow = ElevationMap(replace(c, noise=True), 1, device, .02, Wall())
    shadow.reset(ids[:1])
    shadow.latency[:] = 0
    shadow.observe(torch.tensor([[0., 0., .8]], device=device), yaw[:1], 0)
    assert shadow.shadow_mask[:, shadow.grid[:, 0] > .7].any()
    return {"map_shape": list(cfg.elevation.shape), "original_invalid_fraction": invalid_fraction,
            "randomized_parameters": parameters,
            "component_fractions": fractions, "temporal_checks": "PASS", "history_reset_timeout_checks": "PASS",
            "shadow_wall_check": "PASS", "timeline": timeline, "png": str(out / "stairs_map.png")}


def monitor(out):
    from tools.monitor_loco import assess, parser
    root = out / "monitor_fixtures"
    root.mkdir(exist_ok=True)
    args = parser().parse_args(["--max_hours", "2", "--max_blowups", "20"])
    results = {}
    scenarios = ("nan", "inf", "blowups", "plateau", "hours", "healthy", "missing_eval", "stale_eval", "eval_improves", "level_improves", "partial")
    for name in scenarios:
        run = root / name
        run.mkdir(exist_ok=True)
        (run / "STOP").unlink(missing_ok=True)
        rows = [{"iteration": it, "wall_time": it, "terrain_level": 1., "sim_blowups": 0, "loss": .1} for it in (50, 1000, 4000, 5000)]
        if name in ("nan", "inf"):
            rows[1]["loss"] = float(name)
        if name == "blowups":
            rows[-1]["sim_blowups"] = 25
        if name == "hours":
            rows[-1]["wall_time"] = 8000
        if name == "level_improves":
            rows[-1]["terrain_level"] = 2
        with (run / "train_log.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
        if name == "partial":
            with (run / "train_log.csv").open("a") as f:
                f.write("5050,nan,")
        (run / "eval_loco.csv").unlink(missing_ok=True)
        if name in ("plateau", "stale_eval", "eval_improves", "level_improves"):
            eval_rows = [{"iteration": it, "terrain": "stairs_up", "level": 0, "trials": 8, "success": 0., "map_noise": "on"}
                         for it in ((50, 1000) if name == "stale_eval" else (50, 1000, 4000, 5000))]
            if name == "eval_improves":
                eval_rows[-1]["success"] = .5
            with (run / "eval_loco.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=eval_rows[0]); w.writeheader(); w.writerows(eval_rows)
        result = assess(run, args)
        stopped = (run / "STOP").exists()
        assert stopped == (name in ("nan", "inf", "blowups", "plateau", "hours")), (name, result)
        results[name] = result
    # Preserve a manual STOP and prove CLI execution does not import torch/CUDA.
    (root / "healthy" / "STOP").write_text("manual request\n")
    assess(root / "healthy", args)
    assert (root / "healthy" / "STOP").read_text() == "manual request\n"
    code = ("import runpy,sys; sys.argv=" + repr(["tools/monitor_loco.py", "--root", str(root), "--output", str(root / "monitor.json"),
                                                   "--once", "--max_hours", "2", "--max_blowups", "20"]) +
            "; runpy.run_path('tools/monitor_loco.py',run_name='__main__'); assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True)
    return {"scenarios": results, "cpu_only_cli": "PASS", "preserves_manual_stop": "PASS"}


def lifecycle(out, checkpoint, device):
    """Exercise actual runner STOP/resume and distillation -> PPO transfer."""
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run = out / "resume_distill"
    run.mkdir(exist_ok=True)
    (run / "STOP").write_text("P3 lifecycle test\n")

    def execute(name, arguments):
        with (out / (name + ".log")).open("w") as f:
            subprocess.run([sys.executable, "supermimic/train_student.py", "--device", device,
                            "--num_envs", "64", "--log_every", "1", *arguments], stdout=f, stderr=subprocess.STDOUT, check=True)

    target = source["iter"] + 1
    execute("stop_distill", ["--resume", str(checkpoint), "--iterations", str(target), "--log_dir", str(run)])
    stopped = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert stopped["iter"] == source["iter"]
    for k, v in source["student_state_dict"].items():
        torch.testing.assert_close(stopped["student_state_dict"][k], v, rtol=0, atol=0)
    (run / "STOP").unlink()
    execute("resume_distill", ["--resume", str(run / "last.pt"), "--iterations", str(target), "--log_dir", str(run)])
    resumed = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert resumed["iter"] == target and resumed["common_step_counter"] == source["common_step_counter"] + 24
    assert any(not torch.equal(v, resumed["student_state_dict"][k]) for k, v in source["student_state_dict"].items() if "mlp" in k)
    for k, v in source["teacher_state_dict"].items():
        torch.testing.assert_close(resumed["teacher_state_dict"][k], v, rtol=0, atol=0)
    from supermimic.student import load_student
    from tensordict import TensorDict
    actor = load_student(source)
    obs = TensorDict({"policy_history": torch.randn(4, actor.history_dim), "elevation_map": torch.randn(4, actor.map_dim),
                      "critic": torch.randn(4, 287), "height_scan": torch.randn(4, 187), "policy": torch.randn(4, 96)}, [4])
    with torch.no_grad():
        actions = actor(obs)
        obs["critic"].fill_(float("nan")); obs["height_scan"].fill_(float("nan")); obs["policy"].fill_(float("nan"))
        torch.testing.assert_close(actor(obs), actions, rtol=0, atol=0)
        assert torch.isfinite(actions).all()
    ppo = out / "finetune"
    execute("finetune", ["--stage", "ppo", "--student", str(checkpoint), "--iterations", "2", "--log_dir", str(ppo)])
    init = torch.load(ppo / "ckpt_000000.pt", map_location="cpu", weights_only=False)
    for k, v in source["student_state_dict"].items():
        torch.testing.assert_close(init["actor_state_dict"][k], v, rtol=0, atol=0)
    for k, v in source["teacher_critic_state_dict"].items():
        torch.testing.assert_close(init["critic_state_dict"][k], v, rtol=0, atol=0)
    (ppo / "STOP").write_text("PPO STOP test\n")
    execute("stop_ppo", ["--stage", "ppo", "--resume", str(ppo / "last.pt"), "--iterations", "3", "--log_dir", str(ppo)])
    assert torch.load(ppo / "last.pt", map_location="cpu", weights_only=False)["iter"] == 2
    (ppo / "STOP").unlink()
    execute("resume_ppo", ["--stage", "ppo", "--resume", str(ppo / "last.pt"), "--iterations", "3", "--log_dir", str(ppo)])
    final = torch.load(ppo / "last.pt", map_location="cpu", weights_only=False)
    assert final["iter"] == 3 and final["total_blown"] == 0
    assert final["runner_cfg"]["obs_groups"] == {"actor": ["policy_history", "elevation_map"], "critic": ["critic"]}
    return {"distill_stop_iteration": stopped["iter"], "distill_resume_iteration": resumed["iter"],
            "ppo_resume_iteration": final["iter"], "teacher_unchanged": True, "privileged_input_isolation": "PASS",
            "actor_normalizer_critic_transfer": "PASS", "ppo_blowups": final["total_blown"], "ppo_checkpoint": str(ppo / "last.pt")}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("maps", "monitor", "lifecycle"), required=True)
    p.add_argument("--checkpoint", type=Path, default=Path("logs/p3_student/ckpt_000100.pt"))
    p.add_argument("--out", type=Path, default=Path("logs/p3_checks"))
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "maps":
        result = maps(args.out, args.device)
    elif args.mode == "monitor":
        result = monitor(args.out)
    else:
        result = lifecycle(args.out, args.checkpoint, args.device)
    (args.out / (args.mode + ".json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
