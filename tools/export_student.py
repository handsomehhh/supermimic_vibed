"""Export a P3 actor + normalizer to TorchScript/ONNX and verify both on CPU."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import onnx
from onnx.reference import ReferenceEvaluator
from tensordict import TensorDict

from supermimic.elevation_map import ElevationCfg
from supermimic.robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list, sim_model
from supermimic.student import is_student, load_student


def specification(ck, actor):
    cfg = ck["env_cfg"]
    elevation = ElevationCfg(**cfg["elevation"])
    nx, ny = elevation.shape
    history = cfg["history_length"]
    fields = [("angular_velocity", 0, 3, "rad/s", "base/body frame"),
              ("projected_gravity", 3, 6, "unit vector", "world down expressed in base frame"),
              ("velocity_command", 6, 9, "m/s, m/s, rad/s", "base vx, base vy, yaw rate"),
              ("joint_position_minus_default", 9, 38, "rad", "joint coordinates"),
              ("joint_velocity", 38, 67, "rad/s", "joint coordinates"),
              ("previous_action", 67, 96, "dimensionless", "previous applied action after [-10,10] clamp")]
    model = sim_model(collision=cfg["collision"])
    return {"format_version": 1, "checkpoint": ck.get("iter"), "control_rate_hz": 50, "control_dt_s": .02,
            "input": {"name": "observations", "dtype": "float32", "shape": ["batch", actor.obs_dim],
                      "order": ["policy_history", "map_heights", "map_validity"],
                      "history": {"length": history, "frame_size": 96, "order": "oldest to newest, flattened frame-major",
                                  "slice": [0, history * 96], "sample_period_s": .02,
                                  "reset": "repeat first measured frame across all slots; previous_action=0",
                                  "fields": [{"name": n, "slice": [a, b], "units": u, "frame": f} for n, a, b, u, f in fields]},
                      "map": {"height_slice": [history * 96, history * 96 + nx * ny],
                              "validity_slice": [history * 96 + nx * ny, actor.obs_dim],
                              "shape_xy": [nx, ny], "size_m": list(elevation.size), "resolution_m": elevation.resolution,
                              "forward_offset_m": elevation.forward_offset,
                              "cell_center_start_xy_m": [-elevation.size[0]/2 + elevation.forward_offset + elevation.resolution/2,
                                                         -elevation.size[1]/2 + elevation.resolution/2],
                              "flatten_order": "x outer/increasing, y inner/increasing; all heights then all masks",
                              "frame": "gravity aligned; x forward at acquisition yaw, y left, z up",
                              "height_units": "m", "height_formula": "terrain_z - base_z at acquisition; no standing-height subtraction",
                              "validity": "1=valid, 0=invalid", "invalid_height": elevation.neutral,
                              "timing": "hold complete acquisition-frame map between refreshes; no motion compensation",
                              "startup": "seed pose ring with current pose; no data from prior episode",
                              "training_noise": asdict(elevation)}},
            "normalization": {"included_in_model": True, "formula": "history: (x - mean) / (std + 0.01); map: identity",
                              "applies_to": "proprioceptive history only; heights remain meters, mask remains 0/1; do not normalize externally"},
            "output": {"name": "actions", "dtype": "float32", "shape": ["batch", 29], "units": "dimensionless",
                       "joint_order": G1_JOINT_NAMES, "action_clip": [-10, 10],
                       "default_joint_position_rad": default_joint_pos_list(),
                       "action_scale_rad": [action_scale(n) for n in G1_JOINT_NAMES],
                       "joint_lower_rad": model.jnt_range[1:, 0].tolist(),
                       "joint_upper_rad": model.jnt_range[1:, 1].tolist(),
                       "target_formula": "clip(default + scale * clip(action,-10,10), joint_lower, joint_upper)",
                       "control": "PD joint targets held for four 5 ms simulation substeps; deploy at 50 Hz"}}


def export(checkpoint, out):
    torch.set_num_threads(4)
    torch.manual_seed(123)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not is_student(ck):
        raise ValueError("Expected a P3 student checkpoint")
    actor = load_student(ck)
    model = actor.as_jit().eval()
    out.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(str(out / "student.ts"))
    scripted = torch.jit.load(str(out / "student.ts")).eval()
    dummy = torch.zeros(1, actor.obs_dim)
    torch.onnx.export(model, (dummy,), str(out / "student.onnx"), dynamo=False, opset_version=18,
                      input_names=["observations"], output_names=["actions"],
                      dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}})
    graph = onnx.load(str(out / "student.onnx"))
    onnx.checker.check_model(graph)
    reference = ReferenceEvaluator(graph)
    results = []
    with torch.no_grad():
        for batch in (1, 7, 32):
            x = actor.obs_normalizer._mean + torch.randn(batch, actor.obs_dim) * actor.obs_normalizer._std
            cells = actor.map_dim // 2
            mask = torch.randint(0, 2, (batch, cells)).float()
            x[:, -cells:] = mask
            x[:, actor.history_dim:actor.history_dim + cells].masked_fill_(mask == 0, ck["env_cfg"]["elevation"]["neutral"])
            obs = TensorDict({"policy_history": x[:, :actor.history_dim], "elevation_map": x[:, actor.history_dim:]}, [batch])
            expected = actor(obs)
            jit = scripted(x)
            onnx_out = reference.run(None, {"observations": x.numpy()})[0]
            torch.testing.assert_close(jit, expected, atol=2e-5, rtol=2e-5)
            np.testing.assert_allclose(onnx_out, expected.numpy(), atol=2e-5, rtol=2e-5)
            results.append({"batch": batch, "torchscript_max_abs": float((jit - expected).abs().max()),
                            "onnx_max_abs": float(np.max(np.abs(onnx_out - expected.numpy())))})
    spec = specification(ck, actor)
    spec["verification"] = {"backend": "onnx.reference.ReferenceEvaluator (CPU)", "atol": 2e-5, "rtol": 2e-5, "batches": results}
    (out / "student_spec.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps(spec["verification"], indent=2))
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
