"""P2.5 synthetic semantics, GPU contact validation, and old-default regression."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.types import vec5

from supermimic.loco_env import LocoCfg, LocoEnv, setup_device
from supermimic.stairs import body_contact_vectors, clearance_cost, nearest_edge_heading, stumble_cost
from supermimic.terrain import HeightSampler, Terrain
from supermimic.train_ppo import add_stairs_args, apply_stairs_args, cfg_from_dict, cfg_to_dict
from supermimic.diagnose_stairs import summarize


def semantics(device):
    terrain = Terrain(np.array([[0, 0, .2, .2], [0, 0, .2, .2]], dtype=np.float32),
                      np.zeros((1, 1, 3)), np.zeros((1, 1)), .1, (0., 0.))
    sample = HeightSampler(terrain, device)
    xy = torch.tensor([[[.1, 0], [.2, 0]]], device=device)
    gap = .3 - sample.sample(xy)
    cost = clearance_cost(gap, torch.ones(1, 2, 2, device=device) / np.sqrt(2),
                          torch.zeros(1, 2, dtype=torch.bool, device=device),
                          torch.tensor([[.6, 0., 0.]], device=device), .1)
    torch.testing.assert_close(gap, torch.tensor([[.3, .1]], device=device))
    torch.testing.assert_close(cost, torch.tensor([.2], device=device))
    zeros = torch.zeros(1, 3, device=device)
    contact = torch.ones(1, 2, dtype=torch.bool, device=device)
    assert clearance_cost(gap, xy, contact, zeros + .6, .1).item() == 0
    assert clearance_cost(gap, xy, ~contact, zeros, .1).item() == 0
    force = torch.tensor([[[31., 0., 10.], [0., 0., 100.]],
                          [[30., 0., 10.], [0., 0., 100.]],
                          [[0., 0., 0.], [0., 0., 0.]],
                          [[0., 31., -10.], [0., 0., 0.]]], device=device)
    assert stumble_cost(force).tolist() == [1, 0, 0, 1]
    # Rectangular tile, all four edges, tie resolved by current facing direction.
    xy = torch.tensor([[1.5, 0.], [0., 2.5], [-1.5, 0.], [0., -2.5], [0., 0.]], device=device)
    heading = nearest_edge_heading(xy, torch.zeros_like(xy), xy.new_tensor([2., 3.]),
                                   xy.new_tensor([0., 0., 0., 0., np.pi]))
    torch.testing.assert_close(heading, xy.new_tensor([0, np.pi / 2, np.pi, -np.pi / 2, np.pi]))
    parser = argparse.ArgumentParser()
    add_stairs_args(parser)
    preset = apply_stairs_args(LocoCfg(), parser.parse_args(["--preset", "stairs_v1"]))
    assert preset.foot_clearance_weight == -.25 and preset.outward_fraction == .5
    overridden = apply_stairs_args(cfg_from_dict(cfg_to_dict(preset)), parser.parse_args(
        ["--preset", "stairs_v1", "--foot_stumble_weight", "0", "--outward_fraction", "0"]))
    assert overridden.foot_stumble_weight == 0 and overridden.outward_fraction == 0
    saved = cfg_to_dict(preset)
    assert cfg_to_dict(apply_stairs_args(cfg_from_dict(saved), parser.parse_args([]))) == saved
    return {"synthetic_clearance_stumble_edges_cli": "PASS"}


def synthetic_vectors(device):
    with wp.ScopedDevice(device):
        # Last entry is inactive despite containing nonzero forces. One active
        # contact is on geom 0 to check signs; the other is on geom 1.
        nacon = wp.array([2], dtype=int)
        world = wp.array([0, 1, 0], dtype=int)
        geom = wp.array([[0, 1], [1, 0], [0, 1]], dtype=wp.vec2i)
        address = wp.array([[0, 1, 2, 3]] * 3, dtype=int)
        force = wp.array([[80., 20., 60., 40.]] * 2, dtype=float)
        bodies = wp.array([0, 1], dtype=int)
        frame = wp.array([[[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], np.eye(3), np.eye(3)], dtype=wp.mat33)
        friction = wp.array([[.5] * 5] * 3, dtype=vec5)
        out = wp.zeros((2, 2, 3), dtype=float)
        wp.launch(body_contact_vectors, dim=3, inputs=[nacon, world, geom, address, force, bodies, frame, friction, out])
        torch.testing.assert_close(wp.to_torch(out)[:, 1], torch.tensor([[7.5, 2.5, 50.], [-50., -7.5, -2.5]], device=device))
    return {"synthetic_vector_rotation_friction_sign_and_active_count": "PASS"}


def diagnostic_masks():
    trace = {"alive": np.array([[True, True], [True, True], [False, False]]),
             "valid": np.array([[True, True], [True, False], [True, True]]),
             "falls": np.array([[False, False], [True, True], [False, False]]),
             "foot_force": np.zeros((3, 2, 2, 3)), "shin_normal": np.zeros((3, 2, 2)),
             "qpos": np.zeros((3, 2, 36)), "origins": np.zeros((2, 3)), "start_xy": np.zeros((2, 2)),
             "time": np.array([.02, .04, .06]), "base_height": np.ones((3, 2)) * .75,
             "tilt_deg": np.zeros((3, 2))}
    trace["base_height"][1, 0] = .3
    trace["shin_normal"][1, 0, 0] = 25
    trace["shin_normal"][2] = 1000  # Post-reset data must not contaminate the result.
    trace["qpos"][2, :, 0] = 100
    rows = summarize(trace, [7, 9], 1, .02, 3.)
    assert rows[0]["outcome"] == "fall" and rows[0]["peak_shin_normal_N"] == 25
    assert rows[0]["termination_triggers"] == ["base_height_below_0.35_m"]
    assert rows[0]["max_distance"] == 0 and rows[1]["outcome"] == "invalid"
    return {"diagnostic_first_fall_and_invalid_masks": "PASS"}


def gpu_semantics(device):
    torch.manual_seed(42)
    cfg = LocoCfg(randomization=False, observation_noise=False, contact_vectors=True,
                  foot_clearance_weight=-.25, foot_stumble_weight=-1., outward_fraction=1.,
                  standing_fraction=1.)
    env = LocoEnv(64, cfg, device)
    pyramid = (env.kinds >= 2) & (env.kinds <= 5)
    assert torch.equal(env.outward_mode, pyramid)
    assert not env.standing[pyramid].any() and env.standing[~pyramid].all()
    assert (env.cmd[pyramid, 0] - .3).abs().max() < 1e-6
    assert env.cmd[pyramid, 1:].abs().max() < 1e-6
    chosen = env.ids[pyramid][:4]
    env.qpos[chosen, :2] = env.origins[env.levels[chosen], env.columns[chosen], :2] + env.qpos.new_tensor(
        [[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    target = env.qpos.new_tensor([0, np.pi / 2, np.pi, -np.pi / 2])
    yaw = target + np.pi
    env.qpos[chosen, 3], env.qpos[chosen, 6] = (yaw / 2).cos(), (yaw / 2).sin()
    env.resample_commands(chosen)
    torch.testing.assert_close(env.heading[chosen], target)
    assert (env.cmd[chosen, 0] == 0).all() and (env.cmd[chosen, 1] == 0).all()
    assert (env.cmd[chosen, 2].abs() == .5).all()
    # Commands warm up to the requested outward speed without changing vy.
    env.common_step_counter = 8000 * 24
    env.reset(env.ids)
    assert (env.cmd[pyramid, 0] - .6).abs().max() < 1e-6
    env.cfg.outward_fraction = .5
    count = 0
    for _ in range(40):
        env.resample_commands(env.ids)
        count += int(env.outward_mode.sum())
    share = count / (int(pyramid.sum()) * 40)
    assert .44 < share < .56
    # Stand still for contact checks, no resampling within this window.
    env.cfg.standing_fraction = 1.
    env.cfg.outward_fraction = 0.
    env.reset(env.ids)
    action = torch.zeros(64, 29, device=device)
    for _ in range(25):
        obs, reward, _, extras = env.step(action)
        assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
    foot_z = env.body_force_vector[:, env.feet, 2].sum(-1)
    normal = env.body_force[:, env.feet].sum(-1)
    # CCD normals on the large FP32 hfield need not be exactly vertical.
    torch.testing.assert_close(foot_z, normal, atol=.01, rtol=.01)
    weight = float(env.mjm.body_mass.sum() * 9.81)
    assert ((foot_z - weight).abs() < weight * .2).all()
    for side in ("left", "right"):
        assert env.nonfeet[env.mjm.body(side + "_knee_link").id]
    points, ground, gap = env.foot_clearance()
    assert points.shape[:2] == (64, 2) and gap.abs().max() < .01
    # Independent oracle: CPU MuJoCo pyramid decoder, gathering each constraint
    # address. This installed Warp version's public contact_force assumes the
    # four addresses are consecutive, but its GPU constraint builder uses atomics.
    with wp.ScopedDevice(device):
        c = env.data.contact
        ncon = int(wp.to_torch(env.data.nacon)[0])
        contact_ids = wp.array(np.arange(ncon), dtype=int)
        force = wp.zeros(ncon, dtype=wp.spatial_vector)
        mjw.contact_force(env.model, env.data, contact_ids, True, force)
        out = wp.zeros((64, env.mjm.nbody, 3), dtype=float)
        wp.launch(body_contact_vectors, dim=env.data.naconmax,
                  inputs=[env.data.nacon, c.worldid, c.geom, c.efc_address, env.data.efc.force,
                          env.model.geom_bodyid, c.frame, c.friction, out])
        expected = torch.zeros_like(wp.to_torch(out)).flatten(0, 1)
        geom_ids = wp.to_torch(c.geom)[:ncon].long()
        world = wp.to_torch(c.worldid)[:ncon].long()
        bodies = wp.to_torch(env.model.geom_bodyid)[geom_ids]
        addresses = wp.to_torch(c.efc_address)[:ncon, :4].long()
        pyramid = torch.where(addresses >= 0, wp.to_torch(env.data.efc.force)[world[:, None], addresses.clamp(min=0)], 0)
        pyramid_np = pyramid.cpu().numpy().astype(np.float64)
        mu = wp.to_torch(c.friction)[:ncon].cpu().numpy().astype(np.float64)
        # Use the binding's full spatial-force storage, padding unused friction
        # directions with zero pyramid forces.
        padded = np.pad(pyramid_np, ((0, 0), (0, 6)))
        decoded = np.zeros((ncon, 6))
        for i in range(ncon):
            mujoco.mju_decodePyramid(decoded[i], padded[i], mu[i])
        local = torch.tensor(decoded[:, :3], device=device, dtype=torch.float32)
        world_force = (local[:, None] @ wp.to_torch(c.frame)[:ncon]).squeeze(1)
        for k, sign in ((0, -1), (1, 1)):
            mask = bodies[:, k] > 0
            expected.index_add_(0, world[mask] * env.mjm.nbody + bodies[mask, k],
                                world_force[mask] * sign * .25)
        torch.testing.assert_close(wp.to_torch(out).flatten(0, 1), expected, atol=1e-4, rtol=1e-5)
        # Independently project each solved constraint through its translation
        # Jacobian. The solver's qfrc_constraint workspace can be cleared later.
        J = wp.to_torch(env.data.efc.J)
        ef = wp.to_torch(env.data.efc.force)
        if env.data.efc.J_rownnz.shape[1]:
            nnz, adr, col = [wp.to_torch(a).cpu().numpy() for a in
                             (env.data.efc.J_rownnz, env.data.efc.J_rowadr, env.data.efc.J_colind)]
            nefc = wp.to_torch(env.data.nefc).cpu().numpy()
            j_np, ef_np = J.cpu().numpy(), ef.cpu().numpy()
            root = np.zeros((64, 3))
            for w in range(64):
                for row in range(nefc[w]):
                    a, n = adr[w, row], nnz[w, row]
                    cols = col[w, 0, a:a+n]
                    mask = cols < 3
                    root[w, cols[mask]] += j_np[w, 0, a:a+n][mask] * ef_np[w, row]
            root_force = torch.tensor(root, device=device, dtype=torch.float32)
        else:
            active_rows = torch.arange(ef.shape[1], device=device)[None] < wp.to_torch(env.data.nefc)[:, None]
            root_force = (J[:, :ef.shape[1], :3] * ef[..., None] * active_rows[..., None]).sum(1)
        torch.testing.assert_close(wp.to_torch(out).sum(1) * 4, root_force, atol=.01, rtol=1e-4)
        noncontiguous = int(((addresses[:, 1:] - addresses[:, :-1]) != 1).any(-1).sum())
        api_error = float((world_force - wp.to_torch(force)[:, :3]).abs().max())
    # A few steps followed by stopping cannot promote; expected travel controls
    # demotion, not stair type. All four cases are stairs_down at level 8.
    ids = env.ids[:4]
    env.kinds[ids] = 5
    env.levels[ids] = 8
    env.start_xy[ids] = env.qpos[ids, :2]
    env.qpos[ids, 0] += env.qpos.new_tensor([1.2, 1.2, 4.1, 0.])
    env.command_distance[ids] = env.qpos.new_tensor([12., 2., 12., 0.])
    env.update_curriculum(ids)
    assert env.levels[ids].tolist() == [7, 8, 9, 8]
    return {"gpu_commands_rewards_curriculum": "PASS", "cpu_mujoco_vector_force_oracle": "PASS",
            "generalized_root_force_oracle": "PASS", "noncontiguous_contact_addresses": noncontiguous,
            "installed_warp_contact_force_max_error_N": api_error,
            "outward_sampled_fraction": share, "standing_foot_Fz_N": float(foot_z.mean()),
            "robot_weight_N": weight, "standing_clearance_max_abs_m": float(gap.abs().max()),
            "blowups": int(env.total_blown)}


def regression(device, out, baseline_ref):
    # Read the pre-change module from Git without a checkout or worktree mutation.
    source = subprocess.check_output(["git", "show", f"{baseline_ref}:supermimic/loco_env.py"], text=True)
    path = out.parent / "baseline_loco_env.py"
    path.write_text(source)
    name = "supermimic._p25_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[name] = baseline
    spec.loader.exec_module(baseline)
    old_json = json.dumps(asdict(baseline.LocoCfg()), sort_keys=True)
    new_json = json.dumps(cfg_to_dict(LocoCfg()), sort_keys=True)
    assert old_json.encode() == new_json.encode()

    def snapshot(cls, cfg):
        torch.manual_seed(1234)
        env = cls(64, cfg, device)
        frames = []
        for step in range(6):
            rng = torch.cuda.get_rng_state(device)
            obs = env.get_observations()
            frames.append({"rng": rng, "qvel": env.qvel.cpu().clone(), "last_action": env.last_action.cpu().clone(),
                           "friction": env.friction.cpu().clone(), "qpos": env.qpos.cpu().clone(), "commands": env.cmd.cpu().clone(),
                           **{k: v.cpu().clone() for k, v in obs.items()}})
            action = torch.sin(torch.arange(64 * 29, device=device).reshape(64, 29) + step) * .05
            env.step(action)
        # Force reset/resampling, exercising the unchanged RNG consumption.
        env.reset(env.ids)
        frames.append({"rng": torch.cuda.get_rng_state(device), "qvel": env.qvel.cpu().clone(),
                       "last_action": env.last_action.cpu().clone(), "friction": env.friction.cpu().clone(),
                       "qpos": env.qpos.cpu().clone(), "commands": env.cmd.cpu().clone(),
                       **{k: v.cpu().clone() for k, v in env.get_observations().items()}})
        return frames, env

    old, old_env = snapshot(baseline.LocoEnv, baseline.LocoCfg())
    del old_env
    gc.collect()
    torch.cuda.empty_cache()
    repeated, repeated_env = snapshot(baseline.LocoEnv, baseline.LocoCfg())
    same_code_error = max(float((a["critic"] - b["critic"]).abs().max()) for a, b in zip(old, repeated))
    del repeated_env
    gc.collect()
    torch.cuda.empty_cache()
    new, env = snapshot(LocoEnv, LocoCfg())
    exact = {}
    max_error = {}
    for key in ("qpos", "commands", "policy", "height_scan", "critic"):
        exact[key] = all(torch.equal(a[key], b[key]) for a, b in zip(old, new))
        max_error[key] = max(float((a[key] - b[key]).abs().max()) for a, b in zip(old, new))
        # Initial observations/config and RNG-dependent commands/reset must be
        # byte-identical. Report independent-rollout differences without making
        # a flaky tolerance assertion on nondeterministic GPU contact solving.
        assert torch.equal(old[0][key], new[0][key]), key
    assert torch.equal(old[-1]["commands"], new[-1]["commands"])
    # Stronger observation contract: for exactly the same physical state and
    # random stream, every policy/critic/scan byte must match the pre-change code.
    for frame in old:
        for key in ("qpos", "qvel", "last_action", "friction"):
            getattr(env, key).copy_(frame[key])
        env.cmd.copy_(frame["commands"])
        torch.cuda.set_rng_state(frame["rng"], device)
        for key, value in env.get_observations().items():
            assert torch.equal(value.cpu(), frame[key]), key
    return {"default_config_bytes_identical": True, "initial_observations_bytes_identical": True,
            "all_observations_at_matched_states_bytes_identical": True,
            "repeated_original_code_max_critic_error": same_code_error,
            "six_step_and_reset_byte_equality": exact, "max_abs_error": max_error}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", type=Path, default=Path("logs/p25_checks/unit.json"))
    p.add_argument("--mode", choices=("all", "semantics", "regression"), default="all")
    p.add_argument("--baseline_ref", default="9df2460c8496602d52fa1d7ed0473f79a1a7183a",
                   help="Pre-P2.5 Git revision for default behavior regression")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    setup_device(args.device)
    result = {}
    if args.mode in ("all", "semantics"):
        result.update(semantics(args.device))
        result.update(diagnostic_masks())
        result.update(synthetic_vectors(args.device))
        result.update(gpu_semantics(args.device))
        gc.collect()
        torch.cuda.empty_cache()
    if args.mode in ("all", "regression"):
        result.update(regression(args.device, args.out, args.baseline_ref))
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
