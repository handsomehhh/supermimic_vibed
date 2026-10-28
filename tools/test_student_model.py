"""CPU student architecture, checkpoint and export checks."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tensordict import TensorDict

from supermimic.elevation_map import ElevationCfg, ElevationMap
from supermimic.loco_env import LocoCfg
from supermimic.student import StudentModel, load_student
from supermimic.train_student import finetune_config, student_config
from tools.export_student import export


def observations(cfg, batch):
    class Plane:
        def sample(self, xy):
            return 2 * xy[..., 0] + 5 * xy[..., 1]
    elevation = ElevationMap(cfg.elevation, batch, "cpu", .02, Plane())
    value = elevation.observe(torch.zeros(batch, 3), torch.zeros(batch), 0)
    mask = torch.arange(elevation.cells) % 3 != 0
    value[:, elevation.cells:] = mask.float()
    value[:, :elevation.cells].masked_fill_(~mask, cfg.elevation.neutral)
    return TensorDict({"policy_history": torch.randn(batch, 96 * cfg.history_length), "elevation_map": value}, [batch])


def check(out, encoder, size=(2., 1.2), history=5, latent=64):
    cfg = LocoCfg(history_length=history, elevation=ElevationCfg(enabled=True, noise=False, size=size))
    model_cfg = student_config()
    model_cfg.update(map_latent=latent)
    if encoder == "cnn":
        model_cfg.update(map_encoder="cnn", map_shape=list(cfg.elevation.shape), hidden_dims=[512, 256, 128])
    kwargs = {k: v for k, v in model_cfg.items() if k != "class_name"}
    obs = observations(cfg, 7)
    actor = StudentModel(obs, {"student": ["policy_history", "elevation_map"]}, "student", 29, **kwargs)
    x = torch.cat((obs["policy_history"], obs["elevation_map"]), -1)
    actor.obs_normalizer.update(3 * x + 2)
    normalized = actor.obs_normalizer(x)
    torch.testing.assert_close(normalized[:, actor.history_dim:], obs["elevation_map"], rtol=0, atol=0)
    assert not torch.allclose(normalized[:, :actor.history_dim], obs["policy_history"])
    actor.eval()
    seen = []
    if encoder == "cnn":
        handle = actor.map_encoder.layers[0].register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0]))
    with torch.no_grad():
        expected = actor(obs)
        assert expected.shape == (7, 29) and torch.isfinite(expected).all()
        sequence = TensorDict({k: v.unsqueeze(0).expand(3, -1, -1) for k, v in obs.items()}, [3, 7])
        torch.testing.assert_close(actor(sequence), expected.unsqueeze(0).expand(3, -1, -1))
    if encoder == "cnn":
        handle.remove()
        nx, ny = cfg.elevation.shape
        xs = (torch.arange(nx) + .5) * cfg.elevation.resolution - size[0] / 2 + cfg.elevation.forward_offset
        ys = (torch.arange(ny) + .5) * cfg.elevation.resolution - size[1] / 2
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        mask = (torch.arange(nx)[:, None] * ny + torch.arange(ny)) % 3 != 0
        torch.testing.assert_close(seen[0][0, 0], (2 * xx + 5 * yy).masked_fill(~mask, cfg.elevation.neutral))
        torch.testing.assert_close(seen[0][0, 1], mask.float(), rtol=0, atol=0)
    else:
        assert actor.map_encoder[0].weight.shape == (128, actor.map_dim)
        assert "map_encoder.0.weight" in actor.state_dict() and "map_encoder.2.weight" in actor.state_dict()
    ck = {"env_cfg": asdict(cfg), "runner_cfg": {"policy_type": "student", "stage": "distill", "student": model_cfg},
          "student_state_dict": actor.state_dict()}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "student.pt"
    torch.save(ck, path)
    loaded_ck = torch.load(path, weights_only=False)
    loaded = load_student(loaded_ck)
    with torch.no_grad():
        torch.testing.assert_close(loaded(obs), expected, rtol=0, atol=0)
    assert loaded_ck["runner_cfg"] == ck["runner_cfg"]
    spec = export(path, out / "distill_export")
    ppo_cfg = finetune_config(ck)
    ppo = StudentModel(obs, ppo_cfg["obs_groups"], "actor", 29,
                       **{k: deepcopy(v) for k, v in ppo_cfg["actor"].items() if k != "class_name"})
    missing, unexpected = ppo.load_state_dict(actor.state_dict(), strict=False)
    assert missing and all(k.startswith("distribution.") for k in missing) and not unexpected
    ppo_ck = {"env_cfg": ck["env_cfg"], "runner_cfg": ppo_cfg, "actor_state_dict": ppo.state_dict()}
    torch.save(ppo_ck, out / "ppo.pt")
    with torch.no_grad():
        torch.testing.assert_close(load_student(torch.load(out / "ppo.pt", weights_only=False))(obs), expected, rtol=0, atol=0)
    export(out / "ppo.pt", out / "ppo_export")
    if encoder == "cnn":
        inferred = deepcopy(ck)
        del inferred["runner_cfg"]["student"]["map_shape"]
        with torch.no_grad():
            torch.testing.assert_close(load_student(inferred)(obs), expected, rtol=0, atol=0)
        bad = deepcopy(ck)
        bad["runner_cfg"]["student"]["map_shape"].reverse()
        try:
            load_student(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Transposed checkpoint map geometry accepted")
    return {"parameters": sum(p.numel() for p in actor.parameters()),
            "map_parameters": sum(p.numel() for p in actor.map_encoder.parameters()),
            "shape_xy": list(cfg.elevation.shape), "history_length": history, "map_latent": latent,
            "verification": spec["verification"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("logs/p4_checks/models"))
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    results = {"mlp_default": check(args.out / "mlp", "mlp"),
               "cnn_large": check(args.out / "cnn", "cnn"),
               "cnn_custom": check(args.out / "cnn_custom", "cnn", size=(1.6, .8), history=3, latent=32)}
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
