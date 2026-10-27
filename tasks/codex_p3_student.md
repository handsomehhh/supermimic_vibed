# Codex task P3 — deployable elevation-map student + loco run monitor

Repo: /home/lxz/supermimic_vibed (branch `mjwarp-velocity`). P2 is merged: `supermimic/loco_env.py` (`LocoEnv`,
rsl_rl 5.2 `VecEnv`), `supermimic/train_ppo.py`, `supermimic/eval_loco.py`, collision proxies. Read
`PERCEPTION_MILESTONES.md`, `tasks/codex_p2_report.md` and those files first; keep their style. Python:
`/home/lxz/envs/sm_local/bin/python`; the local RTX 4090 is available for tests. A teacher checkpoint for pipeline
tests: `logs/p2_full/ckpt_000200.pt` (weak, from your P2 smoke run; real teachers are training on the cluster
and will be dropped into `logs/loco/<run>/ckpt_*.pt` with the same format). Only edit files in the repo; no
commits, no network, no new dependencies; keep the motion-tracking paths and existing P2 behavior unchanged
(new features behind options).

## 1. Realistic elevation map observation (`LocoEnv`, option)

A new observation group `elevation_map`: robot-centric, gravity-aligned and yaw-aligned grid (configurable; start
with 2.0 m x 1.2 m at 0.08 m, forward-biased), heights relative to the base, plus a validity mask channel. Model
what an onboard map (G1 MID-360 lidar / depth camera + elevation mapping) delivers, all randomized per env and
per episode with configurable magnitudes:
- per-cell noise (~2 cm) and a slowly drifting vertical offset (±5 cm);
- registration error: xy shift (±5 cm) and yaw error (±3°) of the map relative to the true robot pose;
- latency and refresh: the map is built from the pose/terrain 0–100 ms in the past and refreshed at ~10 Hz
  (held between refreshes);
- missing data: random dropout (5–10%), a blind zone under/near the robot body, and simple shadow/occlusion
  behind tall steps when seen from a sensor at head height (approximate is fine; document it);
- invalid cells filled with a neutral value and flagged in the mask.
Keep `height_scan` (privileged, exact) for the teacher and the critic.

## 2. Student policy and training (`supermimic/train_student.py`)

- Student actor: proprioceptive history (e.g. 5–10 frames of the `policy` group) + `elevation_map` (+ mask) →
  joint actions; a small map encoder (MLP or CNN) is fine. No privileged inputs.
- Training: teacher→student distillation with rsl_rl's `DistillationRunner` (DAgger-style, student drives the
  rollouts) from a P2 teacher checkpoint; then optionally a PPO fine-tune of the student with the privileged
  critic (asymmetric actor–critic). Make both stages selectable. Same logging/checkpoint/STOP/resume conventions
  as `train_ppo.py` (`train_log.csv` every 50 iterations, `ckpt_*.pt`, `last.pt`, STOP file).
- `eval_loco.py` must evaluate student checkpoints too (auto-detect), with map noise on (default) and off, so we
  can report the teacher–student gap per terrain type and level.

## 3. Deployment export

`tools/export_student.py`: export the student actor (with its observation normalizer) to TorchScript and ONNX, plus
a JSON spec of every input (order, units, frames, history length, map grid geometry, mask convention, action
scaling to joint targets, control rate). Verify the exported model matches the PyTorch student numerically.

## 4. Loco run monitor (`tools/monitor_loco.py`)

Runs on the cluster (CPU only) and watches `logs/loco/*/` (or given dirs): reads `train_log.csv` and
`eval_loco.csv`, and writes `STOP` (reason inside) when a run (a) has NaN/inf rows or many blow-ups, (b) shows no
improvement of the mean terrain level and of the eval success rate for a configurable patience (e.g. 3000
iterations) after a minimum (e.g. 4000 iterations), (c) exceeds max hours. It prints one status line per run per
poll and writes `logs/loco/monitor.json`. No GPU use. Make the thresholds CLI options.

## 5. Tests you must run locally and report

- Elevation-map sanity: visualize (PNG) true scan vs noisy map vs mask for a robot standing in front of stairs
  up, with noise on; numbers for dropout/blind-zone fractions; check latency/refresh behavior numerically.
- Distillation smoke test from `logs/p2_full/ckpt_000200.pt`: 1024 envs, ~100 iterations, loss decreasing.
- `eval_loco` on the student checkpoint (noise on/off) runs.
- Export round-trip check.
- `monitor_loco.py` on a synthetic set of logs triggering each rule.

## Report back

`tasks/codex_p3_report.md`: design, noise model with default magnitudes, tests and numbers, and exact commands to
run distillation (and fine-tuning) from a cluster teacher checkpoint. Do not commit.
