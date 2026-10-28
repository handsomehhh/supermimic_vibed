"""Depth geometry, timing, environment, checkpoint/export and GPU throughput checks."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tensordict import TensorDict

from supermimic.depth_camera import DepthCamera, DepthCfg
from supermimic.elevation_map import ElevationCfg
from supermimic.terrain import HeightSampler, Terrain, TerrainCfg, generate


def surface(device, step=False):
    x = np.arange(161) * .05 - 4
    heights = np.broadcast_to(np.where(x >= .8, .4, 0) if step else x * 0, (161, 161)).astype(np.float32).copy()
    return HeightSampler(Terrain(heights, np.empty((0, 0, 3)), np.empty((0, 0), int), .05, (-4., -4.)), device)


def fine_reference(camera, origin, rotation, step=.0005):
    rays = (camera.rays @ rotation[0].T)
    distance = torch.linspace(camera.cfg.near, camera.cfg.far,
                              round((camera.cfg.far - camera.cfg.near) / step) + 1, device=origin.device)
    depths, masks = [], []
    for chunk in rays.split(64):
        points = origin[0] + chunk[:, None] * distance[None, :, None]
        below = points[..., 2] <= camera.sampler.sample(points[..., :2])
        valid = below.any(-1) & ~below[:, 0]
        first = below.int().argmax(-1)
        depths.append(distance[first].masked_fill(~valid, camera.cfg.neutral))
        masks.append(valid)
    return torch.cat(depths)[None], torch.cat(masks)[None]


def camera_checks(device):
    result = {}
    for name in ("flat", "step"):
        camera = DepthCamera(DepthCfg(noise=False), 1, device, .02, surface(device, name == "step"))
        origin = torch.tensor([[0., 0., 1.2]], device=device)
        rotation = camera.mount_rotation[None]
        actual, valid = camera.render(origin, rotation)
        reference, reference_valid = fine_reference(camera, origin, rotation)
        torch.testing.assert_close(valid, reference_valid)
        error = (actual - reference).abs()[valid]
        bound = camera.cfg.coarse_step / 2**camera.cfg.refinement_steps + .0005
        assert error.max() <= bound + 1e-5
        result[name] = {"reference_step_m": .0005, "max_error_m": float(error.max()),
                        "mean_error_m": float(error.mean()), "p99_error_m": float(error.quantile(.99)),
                        "valid_fraction": float(valid.float().mean())}
        if name == "flat":
            dz = (camera.rays @ rotation[0].T)[:, 2]
            exact = -origin[0, 2] / dz
            torch.testing.assert_close(actual[valid], exact[None][valid], atol=bound, rtol=0)
            image = actual.reshape(camera.cfg.shape)
            column = image[:, image.shape[1] // 2]
            column = column[column > 0]
            assert column[0] > column[-1] > 0
            assert torch.all(column[:-1] >= column[1:])
            assert camera.rays[0, 1] > 0 and camera.rays[camera.cfg.resolution[0] - 1, 1] < 0
            up = camera.mount_rotation.T[None]
            assert not camera.render(origin, up)[1].any()
            assert not camera.render(origin * 0, rotation)[1].any()
            assert not camera.render(origin * 100, rotation)[1].any()
    cfg = DepthCfg(mount_position=(0., 0., 0.), depth_noise=0, dropout=(0, 0), edge_dropout=0,
                   position_error=0, rotation_error_degrees=0, refresh_seconds=(1/30, 1/30))
    camera = DepthCamera(cfg, 2, device, .02, surface(device))
    ids = torch.arange(2, device=device)
    camera.reset(ids)
    camera.latency[:] = torch.tensor([0, 3], device=device)
    position = torch.tensor([[0., 0., 1.], [0., 0., 1.]], device=device)
    rotation = torch.eye(3, device=device).expand(2, -1, -1)
    refreshed = 0
    saved = None
    for tick in range(50):
        position[:, 2] = 1 + tick * .002
        value = camera.observe(position, rotation, tick)
        again = camera.observe(position + 10, rotation, tick)
        torch.testing.assert_close(again, value, atol=0, rtol=0)
        if int(camera.refresh_step[0]) == tick:
            refreshed += 1
            past = position.clone()
            past[:, 2] = 1 + .002 * camera.acquisition_step
            expected, valid = camera.render(past, camera.mount_rotation[None].expand(2, -1, -1))
            torch.testing.assert_close(value, torch.cat((expected, valid.float()), -1), atol=0, rtol=0)
            torch.testing.assert_close(camera.acquisition_step, (tick - camera.latency).clamp(min=0))
        else:
            torch.testing.assert_close(value, saved, atol=0, rtol=0)
        saved = value.clone()
    assert refreshed == 30, refreshed
    camera.reset(ids[:1])
    position[0, 2] = 1.5
    reset_value = camera.observe(position, rotation, 49)
    assert camera.acquisition_step[0] == 49
    torch.testing.assert_close(reset_value[1], saved[1], atol=0, rtol=0)
    assert not torch.equal(reset_value[0], saved[0])
    camera.cfg.noise = False
    position[:, 2] += .1
    clean = camera.observe(position, rotation, 50)
    expected, valid = camera.render(position, camera.mount_rotation[None].expand(2, -1, -1))
    torch.testing.assert_close(clean, torch.cat((expected, valid.float()), -1), atol=0, rtol=0)
    noisy_cfg = replace(cfg, noise=True, depth_noise=.005)
    noisy = DepthCamera(noisy_cfg, 64, device, .02, surface(device))
    noisy.reset(torch.arange(64, device=device))
    p = torch.tensor([[0., 0., 1.2]], device=device).expand(64, -1)
    r = torch.eye(3, device=device).expand(64, -1, -1)
    observed = noisy.observe(p, r, 0)[:, :noisy.cells]
    exact, valid = noisy.render(p, noisy.mount_rotation[None].expand(64, -1, -1))
    residual = ((observed - exact) / exact.square())[valid & (exact < 2.8)]
    assert abs(float(residual.std()) - .005) < .0002
    holes = DepthCamera(replace(cfg, noise=True, dropout=(1, 1)), 1, device, .02, surface(device))
    holes.reset(ids[:1])
    assert not holes.observe(p[:1], r[:1], 0).any()
    edge = DepthCamera(replace(cfg, noise=True, edge_dropout=1), 1, device, .02, surface(device, True))
    edge.reset(ids[:1])
    edged = edge.observe(p[:1], r[:1], 0)
    original, mask = edge.render(p[:1], edge.mount_rotation[None])
    assert edged[:, edge.cells:].sum() < mask.sum()
    assert (edged[:, :edge.cells][edged[:, edge.cells:] == 0] == 0).all()
    result.update(orientation_clipping="PASS", refreshes_per_second=refreshed, timing_reset="PASS",
                  noise_sigma_coefficient=float(residual.std()), dropout_edge="PASS")
    return result


def env_checks(device):
    import mujoco
    import warp as wp
    from supermimic.loco_env import LocoCfg, LocoEnv
    cfg = LocoCfg(randomization=False, observation_noise=False, depth=DepthCfg(enabled=True, noise=False))
    env = LocoEnv(32, cfg, device, eval_mode=True)
    obs = env.get_observations()
    assert "elevation_map" not in obs and obs["depth_image"].shape == (32, 1536)
    assert obs["policy_history"].shape == (32, 480)
    again = env.get_observations()
    for key in obs.keys():
        torch.testing.assert_close(obs[key], again[key], atol=0, rtol=0)
    before = obs["policy_history"].reshape(32, 5, 96).clone()
    stepped, _, _, _ = env.step(torch.zeros(32, 29, device=device))
    torch.testing.assert_close(stepped["policy_history"].reshape(32, 5, 96)[:, :-1], before[:, 1:], atol=0, rtol=0)
    saved = stepped.clone()
    env.reset(env.ids[:1])
    reset = env.get_observations()
    torch.testing.assert_close(reset["policy_history"][0].reshape(5, 96), reset["policy"][0].expand(5, -1))
    torch.testing.assert_close(reset["depth_image"][1:], saved["depth_image"][1:], atol=0, rtol=0)
    for key in stepped.keys():
        torch.testing.assert_close(stepped[key], saved[key], atol=0, rtol=0)
    env.max_episode_length = 2
    _, _, _, extras = env.step(torch.zeros(32, 29, device=device))
    assert "depth_image" in extras["terminal_observation"]
    env.reset(env.ids)
    env.qpos[:, env.mjm.joint("waist_pitch_joint").qposadr[0]] = .3
    observed = env.get_observations()["depth_image"]
    d = mujoco.MjData(env.mjm)
    d.qpos[:] = env.qpos[0].cpu().numpy()
    mujoco.mj_kinematics(env.mjm, d)
    actual_R = wp.to_torch(env.data.xmat)[:, env.depth_body].reshape(-1, 3, 3)
    torch.testing.assert_close(actual_R[0].cpu(), torch.tensor(d.xmat[env.depth_body].reshape(3, 3), dtype=torch.float32))
    camera_pose = env.depth.poses[0, 0]
    torch.testing.assert_close(camera_pose[3:].reshape(3, 3), actual_R[0])
    assert not torch.allclose(actual_R[0], torch.eye(3, device=device))
    assert torch.isfinite(observed).all()
    both_env = LocoEnv(4, replace(cfg, elevation=ElevationCfg(enabled=True)), device, eval_mode=True)
    both = both_env.get_observations()
    assert {"depth_image", "elevation_map", "policy_history"} <= set(both.keys())
    return {"depth_only_and_both": "PASS", "history_partial_reset_timeout": "PASS", "articulated_mount": "PASS"}


def model_checks(out):
    from supermimic.loco_env import LocoCfg
    from supermimic.student import StudentModel, load_student
    from supermimic.train_student import student_config, finetune_config
    from supermimic.train_ppo import cfg_from_dict, cfg_to_dict
    from tools.export_student import export
    result = {}
    for size, history in (((32, 24), 5), ((20, 15), 3)):
        cfg = LocoCfg(depth=DepthCfg(enabled=True, resolution=size), history_length=history)
        assert cfg_from_dict(cfg_to_dict(cfg)).depth == cfg.depth
        model_cfg = student_config() | {"depth_shape": list(cfg.depth.shape)}
        obs = TensorDict({"policy_history": torch.randn(7, history * 96),
                          "depth_image": torch.rand(7, 2 * size[0] * size[1])}, [7])
        model = StudentModel(obs, {"student": ["policy_history", "depth_image"]}, "student", 29,
                             **{k: v for k, v in model_cfg.items() if k != "class_name"}).eval()
        x = torch.cat(tuple(obs.values()), -1)
        model.obs_normalizer.update(x + 2)
        torch.testing.assert_close(model.obs_normalizer(x)[:, model.history_dim:], obs["depth_image"], atol=0, rtol=0)
        expected = model(obs).detach()
        sequence = TensorDict({k: v[None].expand(3, -1, -1) for k, v in obs.items()}, [3, 7])
        torch.testing.assert_close(model(sequence), expected[None].expand(3, -1, -1))
        obs["height_scan"] = torch.full((7, 187), float("nan"))
        obs["critic"] = torch.full((7, 287), float("nan"))
        torch.testing.assert_close(model(obs), expected, atol=0, rtol=0)
        ck = {"env_cfg": asdict(cfg), "runner_cfg": {"policy_type": "student", "stage": "distill", "student": model_cfg,
               "obs_groups": {"student": ["policy_history", "depth_image"]}}, "student_state_dict": model.state_dict()}
        path = out / f"{size[0]}x{size[1]}"
        path.mkdir(parents=True, exist_ok=True)
        torch.save(ck, path / "student.pt")
        torch.testing.assert_close(load_student(torch.load(path / "student.pt", weights_only=False))(obs), expected, atol=0, rtol=0)
        spec = export(path / "student.pt", path / "export")
        assert spec["input"]["depth"]["shape_hw"] == list(cfg.depth.shape) and "map" not in spec["input"]
        inferred = deepcopy(ck)
        del inferred["runner_cfg"]["student"]["depth_shape"]
        torch.testing.assert_close(load_student(inferred)(obs), expected, atol=0, rtol=0)
        bad = deepcopy(ck)
        bad["runner_cfg"]["student"]["depth_shape"].reverse()
        try:
            load_student(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Transposed depth_shape accepted")
        ppo_cfg = finetune_config(ck)
        ppo = StudentModel(obs, ppo_cfg["obs_groups"], "actor", 29,
                           **{k: deepcopy(v) for k, v in ppo_cfg["actor"].items() if k != "class_name"})
        missing, unexpected = ppo.load_state_dict(model.state_dict(), strict=False)
        assert missing and all(k.startswith("distribution.") for k in missing) and not unexpected
        ppo_ck = {"env_cfg": asdict(cfg), "runner_cfg": ppo_cfg, "actor_state_dict": ppo.state_dict()}
        torch.save(ppo_ck, path / "ppo.pt")
        torch.testing.assert_close(load_student(ppo_ck)(obs), expected, atol=0, rtol=0)
        export(path / "ppo.pt", path / "ppo_export")
        result[str(size)] = spec["verification"]
    return result


def benchmark(device):
    terrain = generate(TerrainCfg())
    sampler = HeightSampler(terrain, device)
    centers = torch.tensor(terrain.origins.reshape(-1, 3), device=device)
    result = []
    for resolution in ((32, 24), (48, 36), (64, 48)):
        for n in (256, 1024, 4096):
            camera = DepthCamera(DepthCfg(resolution=resolution), n, device, .02, sampler)
            ids = torch.arange(n, device=device)
            positions = torch.tensor([[0., 0., 1.2]], device=device).repeat(n, 1)
            # Spread origins through the full terrain and all difficulty levels.
            positions += centers[ids % len(centers)]
            rotation = camera.mount_rotation[None].expand(n, -1, -1)
            for _ in range(3):
                camera.render(positions, rotation)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                camera.render(positions, rotation)
            torch.cuda.synchronize()
            render_ms = (time.perf_counter() - start) * 1000 / 20
            camera.reset(ids)
            body_rotation = torch.eye(3, device=device).expand(n, -1, -1)
            for tick in range(10):
                camera.observe(positions, body_rotation, tick)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for tick in range(10, 60):
                camera.observe(positions, body_rotation, tick)
            torch.cuda.synchronize()
            observation_ms = (time.perf_counter() - start) * 1000 / 50
            row = {"resolution_wh": list(resolution), "envs": n, "render_frame_ms": render_ms,
                   "noisy_observation_per_control_step_ms": observation_ms}
            result.append(row)
            print(json.dumps(row), flush=True)
    return result


def monitor_checks(out):
    from tools.monitor_loco import assess, parser
    run = out / "depth_monitor"
    run.mkdir(exist_ok=True)
    (run / "train_log.csv").write_text("iteration,wall_time,terrain_level,sim_blowups,loss\n6,4,0,0,.1\n")
    (run / "eval_loco.csv").write_text("iteration,terrain,level,trials,success,depth_noise\n"
                                       "6,flat,0,1,0.25,on\n6,flat,0,1,0.75,off\n")
    for mode, expected in (("on", .25), ("off", .75)):
        result = assess(run, parser().parse_args(["--eval_depth_noise", mode]))
        assert result["status"] == "running" and not result["reasons"]
        assert result["eval_success"] == expected
    return {"depth_noise_csv_and_filter": "PASS"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--mode", choices=("camera", "env", "models", "benchmark", "monitor", "all"), default="all")
    p.add_argument("--out", type=Path, default=Path("logs/p5_checks"))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.device != "cpu":
        from supermimic.loco_env import setup_device
        setup_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    result = {}
    if args.mode in ("camera", "all"):
        result["camera"] = camera_checks(args.device)
    if args.mode in ("env", "all"):
        result["env"] = env_checks(args.device)
    if args.mode in ("models", "all"):
        result["models"] = model_checks(args.out / "models")
    if args.mode in ("benchmark", "all"):
        result["benchmark"] = benchmark(args.device)
    if args.mode in ("monitor", "all"):
        result["monitor"] = monitor_checks(args.out)
    (args.out / (args.mode + "_" + args.device.replace(":", "_") + ".json")).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
