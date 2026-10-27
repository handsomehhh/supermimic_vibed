"""GPU semantic checks and full-environment throughput for the P2 teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from supermimic.loco_env import LocoCfg, LocoEnv, setup_device
from supermimic.robot import sim_model
from supermimic.train_ppo import LocoRunner, runner_config


def models() -> dict:
    baseline = types.ModuleType("supermimic._p1_robot")
    baseline.__file__ = str(Path(__file__).resolve().parents[1] / "supermimic/robot.py")
    baseline.__package__ = "supermimic"
    exec(subprocess.check_output(["git", "show", "HEAD:supermimic/robot.py"], text=True), baseline.__dict__)
    old, mesh, proxy = baseline.sim_model(), sim_model(), sim_model(collision="proxy")
    serialized = []
    for model in (old, mesh):
        buffer = np.zeros(mujoco.mj_sizeModel(model), dtype=np.uint8)
        mujoco.mj_saveModel(model, buffer=buffer)
        serialized.append(buffer)
    np.testing.assert_array_equal(*serialized)
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_range", "dof_armature",
                  "actuator_gainprm", "actuator_biasprm", "actuator_forcerange"):
        np.testing.assert_array_equal(getattr(mesh, field), getattr(proxy, field))
    active = (proxy.geom_bodyid > 0) & (proxy.geom_contype != 0)
    assert not (proxy.geom_type[active] == mujoco.mjtGeom.mjGEOM_MESH).any()
    table = []
    for body in np.unique(proxy.geom_bodyid[active]):
        ids = np.flatnonzero(active & (proxy.geom_bodyid == body))
        table.append({"body": proxy.body(body).name, "count": len(ids),
                      "type": mujoco.mjtGeom(proxy.geom_type[ids[0]]).name,
                      "size": proxy.geom_size[ids].round(6).tolist()})
    print("default mesh model byte-identical; proxy masses/inertias/actuators identical: PASS", flush=True)
    return {"models": "PASS", "model_bytes": len(serialized[0]), "proxy_table": table}


def semantics(device: str) -> dict:
    torch.manual_seed(42)
    env = LocoEnv(64, LocoCfg(randomization=False, observation_noise=False), device)
    obs = env.get_observations()
    assert [obs[k].shape[-1] for k in ("policy", "height_scan", "critic")] == [96, 187, 287]
    torch.testing.assert_close(obs["policy"], obs["critic"][:, :96])
    torch.testing.assert_close(obs["height_scan"], obs["critic"][:, 96:283])
    for step, bound in ((0, .5), (3000 * 24, .7), (8000 * 24, 1.0)):
        env.common_step_counter = step
        env.resample_commands(env.ids)
        assert env.cmd[:, :2].abs().max() <= bound
        assert env.cmd[:, :2].abs().max() > bound * .8
    env.common_step_counter = 0
    env.reset(env.ids)
    zeros = torch.zeros(64, 29, device=device)
    for _ in range(25):
        obs, r, d, extra = env.step(zeros)
        assert torch.isfinite(r).all() and all(torch.isfinite(v).all() for v in obs.values())
    normal = env.body_force[:, env.feet].sum(-1)
    weight = float(env.mjm.body_mass.sum() * 9.81)
    assert ((normal - weight).abs() < weight * .2).all()
    assert (env.body_force[:, env.nonfeet] == 0).all()
    print(f"contact force: mean feet {normal.mean().item():.3f} N vs weight {weight:.3f} N", flush=True)

    # Forced curriculum transitions: promotion, demotion, standing, top rerandomization.
    ids = env.ids[:4]
    env.levels[ids] = torch.tensor([1, 2, 3, 9], device=device)
    env.start_xy[ids] = env.qpos[ids, :2]
    env.qpos[ids[[0, 3]], 0] += 4.1
    env.command_distance[ids] = torch.tensor([2., 2., 0., 2.], device=device)
    env.update_curriculum(ids)
    assert env.levels[:3].tolist() == [2, 1, 3]
    assert 0 <= int(env.levels[3]) < 10
    env.reset(env.ids)
    torch.testing.assert_close(env.qpos[:, :2], env.origins[env.levels, env.columns, :2])
    env.max_episode_length = 2
    env.step(zeros)
    obs, _, done, extra = env.step(zeros)
    assert done.all() and extra["time_outs"].all() and "terminal_observation" in extra
    assert (env.episode_length_buf == 0).all()
    assert (extra["terminal_observation"]["policy"][:, 38:67].abs().sum(-1) > 0).all()
    env.max_episode_length = 1000

    # Deliberately corrupt one action: rollback, reset, finite observations, invalid flag.
    action = zeros.clone()
    action[0, 0] = float("nan")
    obs, reward, done, extra = env.step(action)
    assert done[0] and not extra["valid"][0] and not extra["time_outs"][0] and reward[0] == 0
    assert all(torch.isfinite(v).all() for v in obs.values()) and env.episode_length_buf[0] == 0

    cfg = runner_config()
    cfg["num_steps_per_env"] = 3
    runner = LocoRunner(env, cfg, device=device)
    alg = runner.alg
    with torch.inference_mode():
        for t in range(3):
            obs = env.get_observations()
            alg.act(obs)
            # Construct known values/returns to check timeout bootstrap and GAE cut.
            alg.transition.values[:] = 2
            rewards = torch.ones(64, device=device)
            valid = torch.ones(64, dtype=torch.bool, device=device)
            done = torch.ones_like(valid)
            valid[0] = t != 1
            if t == 0:
                done[0] = False
            if t == 1:
                rewards[0] = 1000  # Must not leak backwards through the invalid step.
            timeout = torch.zeros_like(valid)
            timeout[1] = t == 2
            extras = {"valid": valid, "time_outs": timeout}
            if t == 2:
                extras["terminal_observation"] = obs
                expected = 1 + alg.gamma * alg.critic(obs)[1, 0]
            alg.process_env_step(obs, rewards, done, extras)
        alg.compute_returns(obs)
        torch.testing.assert_close(alg.storage.returns[2, 1, 0], expected)
        assert alg.storage.advantages[1, 0, 0] == 0
        batches = list(alg._valid_batches(4, 1))
        assert sum(len(b.observations) for b in batches) == 3 * 64 - 1
        torch.testing.assert_close(alg.storage.returns[0, 0, 0], torch.tensor(1 + alg.gamma * 2, device=device))
    loss = alg.update()
    assert all(np.isfinite(v) for v in loss.values())
    small = LocoEnv(1, LocoCfg(scan_shape=(5, 3), scan_size=(.8, .4)), device)
    small_obs, _, _, small_extra = small.step(torch.zeros(1, 29, device=device))
    assert small_obs["height_scan"].shape == (1, 15) and small_obs["critic"].shape == (1, 115)
    assert all(torch.isfinite(v).all() for v in small_extra["log"].values())
    print("observations, contacts, curriculum, timeout bootstrap, guard, invalid PPO exclusion: PASS", flush=True)
    return {"semantic_checks": "PASS", "foot_normal_force": float(normal.mean()), "robot_weight": weight,
            "injected_blowups": int(env.total_blown), "ppo_loss": loss}


def benchmark(collision: str, device: str, n: int, steps: int) -> dict:
    torch.manual_seed(42)
    env = LocoEnv(n, LocoCfg(collision=collision, randomization=False, observation_noise=True, curriculum=False), device)
    action = torch.zeros(n, 29, device=device)
    for _ in range(10):
        env.step(action)
    rates = []
    for _ in range(3):
        env.reset(env.ids)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            obs, rewards, _, _ = env.step(action)
        torch.cuda.synchronize()
        rates.append(n * steps / (time.perf_counter() - start))
        assert torch.isfinite(rewards).all() and all(torch.isfinite(v).all() for v in obs.values())
    result = {"collision": collision, "num_envs": n, "steps": steps, "rates": rates,
              "median_steps_s": float(np.median(rates)), "blowups": int(env.total_blown)}
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("semantics", "benchmark", "models"), default="semantics")
    p.add_argument("--collision", choices=("mesh", "proxy"), default="proxy")
    p.add_argument("--num_envs", type=int, default=4096)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.mode == "models":
        result = models()
    else:
        setup_device(args.device)
        result = semantics(args.device) if args.mode == "semantics" else benchmark(args.collision, args.device, args.num_envs, args.steps)
    args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
