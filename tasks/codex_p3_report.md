P3 implementation and local validation are complete. The required 1,024-environment, 100-iteration distillation run decreased imitation loss by 52.34% between iterations 50 and 100, with zero simulation blow-ups. Both noisy-map and clean-map evaluation, TorchScript/ONNX verification, monitor rules, STOP/resume, and the optional PPO handoff passed. **Terrain traversal is not demonstrated:** the supplied weak teacher and the student both have 0/560 successes. A stronger cluster teacher is still needed to assess the P3 performance milestone.

All Python used `/home/lxz/envs/sm_local/bin/python`; GPU checks used the local RTX 4090. No network, dependency installation, or commits. All generated files and caches are inside this repository. The pre-existing changes to `PERCEPTION_MILESTONES.md` and `tasks/codex_p3_student.md` were left intact. Motion-tracking files are unchanged.

**Implementation and observation contract**

- `supermimic/elevation_map.py` implements `ElevationCfg` and the per-environment map state. `LocoCfg.elevation.enabled` defaults to false, preserving the P2 observation groups and behavior. Enabling it adds `policy_history` and `elevation_map`; the privileged `height_scan` and `critic` remain available for supervision and asymmetric PPO.
- The default map covers 2.0 × 1.2 m with 0.08 m cells: 25 × 15 cell centers, x increasing from -0.46 to 1.46 m and y from -0.56 to 0.56 m. Its boundaries are x=[-0.5, 1.5], y=[-0.6, 0.6]. The frame is gravity aligned, with x along acquisition yaw and y left. Heights are **terrain_z minus acquisition base_z**, in meters; this differs deliberately from the teacher's standing-height-subtracted scan. Flattening uses x outer, y inner; 375 heights precede 375 binary validity values. Invalid heights are zero, with mask=0; valid mask=1.
- History contains five 96-value `policy` frames, oldest to newest, sampled once per 20 ms control tick. Its fields are body angular velocity, projected gravity, velocity command, relative joint position, joint velocity, and previous action. Resets repeat the first measured frame across the history and seed the entire pose ring with the new pose. Repeated observation reads do not advance history, resample the delivered map, or expose data from a previous episode. Returned observations own their storage.
- `supermimic/student.py` supplies the 1,230-input actor. A 750→128→64 ELU map encoder joins the 480-value history, followed by a 544→256→128→29 ELU action network. Only proprioceptive history is empirically normalized. Heights stay in meters and masks stay 0/1. This avoids amplifying previously blind cells when an ideal map makes them valid. An initial per-cell-normalization experiment exposed that problem; it was corrected before the final run reported below.
- `supermimic/train_student.py` provides `--stage distill` and `--stage ppo`. `StudentRunner` inherits rsl-rl 5.2 `DistillationRunner` and the P2 `LocoRunner` loop. Stock rsl-rl Distillation obtains teacher labels on student-driven rollouts; the deterministic student drives every transition. Defaults are 24 steps/env, two learning epochs, MSE, Adam at 3e-4, gradient length 1, and gradient norm limit 1. The teacher and its normalizer stay frozen. The saved teacher critic enables PPO without requiring the original teacher file again.
- PPO initializes the complete student actor and normalizer, adds Gaussian exploration with initial std=0.1, and initializes the privileged critic from the teacher. It uses the existing `ValidPPO` invalid-transition exclusion and terminal-state timeout bootstrap, with initial LR=1e-4 and entropy coefficient=0.001. New stages retain terrain levels and the command-curriculum clock while starting their iteration counter at zero.
- Both stages use the existing CSV loop (default every 50 iterations), atomic `ckpt_NNNNNN.pt`/`last.pt`, STOP checks, and resume behavior. Stage initialization also writes checkpoint zero. Resume restores model/optimizer/normalizer, LR, curriculum levels/counters, RNG state, and accumulated training wall time, starting fresh episodes as in P2. `--iterations` is the total target for that stage. Use a separate directory for PPO.
- `eval_loco.py` auto-detects distilled and PPO student checkpoints. `--map_noise on` is the default; `off` produces exact, fully valid maps at every control tick, and `both` evaluates both modes. Student JSON filenames include `_noise_on`/`_noise_off`; `eval_loco.csv` retains both modes and replaces matching iteration/mode rows on re-evaluation. Teacher output naming remains unchanged. The watcher understands both modes.

**Default map errors**

Episode parameters are sampled separately for each environment. Random cell errors and dropout are sampled at refresh, then held with the complete map.

| Effect | Default model |
|---|---|
| Cell noise | Gaussian; episode standard deviation U[1, 3] cm, centered on 2 cm |
| Vertical drift | Bounded sinusoidal offset ±5 cm; random phase, angular rate U[0.03, 0.15] rad/s, periods about 42–209 s |
| Registration | Episode XY shift independently ±5 cm per axis in the map plane, yaw error ±3° |
| Acquisition latency | Episode delay uniformly selected from 0, 20, 40, 60, 80, 100 ms; delayed xyz/yaw come from a pose ring |
| Refresh | Episode period 4–6 control ticks, 80–120 ms, approximately 10 Hz; first map after reset is immediate |
| Dropout | Episode probability U[5%, 10%], independently sampled per cell at each refresh |
| Body blind zone | Ellipse with default radii 0.30 × 0.22 m, scaled by episode U[0.8, 1.2] |
| Sensor height | Base z + 0.65 m, with episode ±10% variation |
| Shadow | 16 intermediate terrain samples along each sensor-to-cell line; mask the cell if any terrain sample exceeds that line by more than 2 cm |

Terrain is static, so storing delayed poses is sufficient to sample past terrain observations. Delivered maps stay in the **acquisition frame**, without compensation for subsequent motion, and heights remain relative to the acquisition base. Holding between refreshes can add up to 100 ms to the configured acquisition delay. Startup repeats the initial pose rather than reading nonexistent historical data. The shadow test approximates heightfield visibility; it does not model overhangs, dynamic obstacles, beam patterns, or a full lidar/depth reconstruction pipeline. All geometry and error settings are configurable through `ElevationCfg`, or a JSON override passed to `--map_config` for new distillation runs.

**Deployment artifacts**

`tools/export_student.py` loads only the actor on CPU, writes `student.ts`, `student.onnx` (opset 18, dynamic batch), and `student_spec.json`, and verifies both exported models against the original TensorDict-based PyTorch actor. ONNX verification uses the already installed `onnx.reference.ReferenceEvaluator`; ONNX Runtime is not installed and was not added. The JSON describes every input slice, units/frame/order, history/reset rules, grid centers, map/mask convention, timing, included normalization, ordered joint names, default positions, per-joint scales/limits, action clipping, joint-target formula, and 50 Hz control rate. The exported actor is stateless; deployment supplies history and the held map according to this contract.

Final distilled artifacts: `logs/p3_student/export/{student.ts,student.onnx,student_spec.json}`. PPO artifacts were independently exported under `logs/p3_checks/finetune/export/`.

**CPU monitor**

`tools/monitor_loco.py` uses only the standard library. It polls `logs/loco/*/` or explicit directories, prints one line per run, and atomically writes `logs/loco/monitor.json` by default. It creates a reasoned STOP request for completed nonfinite/malformed metric rows, too many cumulative blow-ups within a rolling iteration window, simultaneous level and evaluation stagnation after the minimum iteration, or excessive elapsed hours. Hours include logged training time plus time since the latest CSV flush, allowing stalled runs to time out. Existing STOP reasons are preserved.

Plateau detection requires both metrics to have failed to improve by their configured deltas for the patience interval. It needs evaluation coverage over that interval and rejects stale/missing evaluation evidence. Evaluation success is trial-weighted across terrain/level cases; noisy student evaluation is selected by default. It ignores an unfinished trailing CSV write. Thresholds, evaluation mode/maximum lag, paths, and polling interval are CLI options; `--once` runs a single poll. The monitor requests training to stop; it does not kill a process or infer that a STOP request has already been honored.

**Local evidence**

The map PNG is `logs/p3_checks/stairs_map.png`: exact terrain heights on the map grid, noisy heights, and validity for a robot positioned on the rising side of level-9 stairs_up. Across 128 robots at that pose, dropout was **7.7792%**, blind-zone coverage **8.7708%**, and their union **15.8583%**. No shadow occurred on this visible ascending staircase; a separate tall-step/wall fixture verified masking behind an occluder. Measured episode noise sigma ranged 1.0558–2.9953 cm, registration reached 4.9897 cm / 2.9985°, latency covered 0–5 ticks, and refresh covered 4–6 ticks.

The numerical temporal test moved the base up by 0.1 m/tick above flat terrain. With a three-tick delay and five-tick refresh, the delivered height was -1.0 m at ticks 0–4, -1.2 m at ticks 5–9, -1.7 m at ticks 10–14, and -2.2 m at tick 15, matching acquisition ticks 0, 2, 7, and 12. A second environment independently refreshed every four ticks with zero delay. Partial resets, history ordering, repeated reads, terminal observations, held values, and clean-map refresh all passed. Full values are in `logs/p3_checks/maps.json`.

Final distillation used `logs/p2_full/ckpt_000200.pt`, 1,024 environments, 100 iterations, 24 steps/env, and seed 42: **2,457,600 transitions**, 65.45 s in the training loop excluding setup. The teacher actor and normalizer remained tensor-for-tensor identical to the P2 checkpoint. Student weights and every training CSV numeric field are finite.

| Iteration | Behavior MSE | Mean step reward | Episode return | Episode length | Mean terrain level | Blow-ups |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.00327410 | 0.082015 | 38.499 | 450.186 | 0.000 | 0 |
| 100 | 0.00156035 | 0.089115 | 71.671 | 805.121 | 0.000 | 0 |

The final student ran 560 evaluation trials per mode: seven terrain types × ten levels × eight headings, 8 s at ±0.6 m/s. Both modes had **zero falls, zero blow-ups, and zero successes**. The existing matching P2 evaluation had 12 falls, zero blow-ups, and zero successes. Mean maximum displacement for the student was only 0.013–0.056 m depending on terrain/mode, versus the 3 m success threshold. These policies mainly stand; zero success gap does not establish terrain competence.

| Terrain | Teacher XY tracking error (m/s) | Student, noise on | Student, noise off |
|---|---:|---:|---:|
| flat | 0.601165 | 0.600626 | 0.600099 |
| rough | 0.601204 | 0.600721 | 0.600185 |
| slope_up | 0.600588 | 0.600230 | 0.599842 |
| slope_down | 0.595573 | 0.595373 | 0.595855 |
| stairs_up | 0.600019 | 0.600104 | 0.599751 |
| stairs_down | 0.595769 | 0.595027 | 0.595276 |
| boxes | 0.636383 | 0.600180 | 0.599796 |

Per-terrain/per-level comparison, including success gaps, falls, distance, and tracking error: `logs/p3_checks/teacher_student_gap.csv`. Original student results are the 140 unique rows in `logs/p3_student/eval_loco.csv` and the two `ckpt_000100_eval_noise_*.json` files. The teacher column uses the existing eight-trial P2 evaluation at `logs/p2_full/ckpt_000200_eval.json`.

| Additional check | Result |
|---|---|
| Export round trip, batches 1/7/32 | Distilled actor: TorchScript max absolute error 0; ONNX max 5.96046e-7. PPO actor: TorchScript 0; ONNX max 3.57628e-7. Tolerance 2e-5 absolute/relative. Both ONNX graphs passed checker validation. |
| Distillation STOP/resume | STOP saved iteration 100 with identical student weights; removing STOP and resuming reached 101, changed actor weights, retained teacher weights, and advanced the command clock by 24. Resume also exercised 1,024→64 environment remapping. |
| PPO handoff/STOP/resume | Initial actor, normalizer, and teacher critic copied exactly. Two iterations, STOP at 2, then resume to 3; zero blow-ups and finite metrics. |
| PPO checkpoint evaluation | Both map modes auto-detected and ran 70 trials each; zero falls/blow-ups/successes. |
| Privileged-input isolation | Replacing `policy`, `height_scan`, and `critic` with NaNs left student outputs exactly unchanged and finite. Actor groups are exclusively history/map; PPO critic group is exclusively `critic`. |
| Monitor fixtures | NaN, inf, blow-up, plateau, and max-hours scenarios each wrote STOP. Healthy, missing-eval, stale-eval, improving-eval, improving-level, and partial-write scenarios did not. Manual STOP preserved. CLI subprocess asserted that `torch` was never imported. |
| P2 regression | Existing `tools/test_loco.py --mode semantics` passed observation dimensions, contact loads, curriculum, terminal bootstrap, numerical guard, invalid PPO exclusion, and configurable scans. |
| Static/artifact checks | Nine changed/new Python files parsed; `git diff --check` passed; final checkpoints/CSV metrics finite; teacher actor/normalizer unchanged; 140 distinct iteration/terrain/level/noise evaluation rows. |

Evidence is under `logs/p3_checks/`, especially `maps.json`, `lifecycle.json`, `monitor.json`, `p2_regression.json`, `eval_summary.json`, and `final_validation.json`. The initial discarded normalization experiment is retained in `logs/p3_student_initial/`; its results are not the final results above.

**Commands**

Run from the repository root. These use the requested interpreter; the teacher path below is an example cluster checkpoint in the P2 format and should name the chosen trained teacher. Distillation and PPO use separate log directories and stage iteration counters.

```bash
export TMPDIR="$PWD/.cache/tmp"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"

/home/lxz/envs/sm_local/bin/python supermimic/train_student.py \
  --stage distill --teacher logs/loco/full_s0/ckpt_010000.pt \
  --num_envs 4096 --history_length 5 --iterations 10000 --seed 42 \
  --map_noise on --save_every 1000 --log_every 50 \
  --log_dir logs/loco/student_full_s0

/home/lxz/envs/sm_local/bin/python supermimic/train_student.py \
  --stage ppo --student logs/loco/student_full_s0/ckpt_010000.pt \
  --num_envs 4096 --iterations 20000 --seed 42 \
  --map_noise on --save_every 1000 --log_every 50 \
  --log_dir logs/loco/student_full_s0_ppo
```

To resume after a monitor/operator stop, first review the STOP reason and remove that run's STOP file when continuation is intended. The commands below restore the saved stage configuration rather than rebuilding it from new observation options.

```bash
/home/lxz/envs/sm_local/bin/python supermimic/train_student.py \
  --stage distill --resume logs/loco/student_full_s0/last.pt \
  --num_envs 4096 --iterations 10000 --log_dir logs/loco/student_full_s0

/home/lxz/envs/sm_local/bin/python supermimic/train_student.py \
  --stage ppo --resume logs/loco/student_full_s0_ppo/last.pt \
  --num_envs 4096 --iterations 20000 --log_dir logs/loco/student_full_s0_ppo

/home/lxz/envs/sm_local/bin/python supermimic/eval_loco.py \
  --run_dir logs/loco/student_full_s0 --watch --map_noise both --trials 8

/home/lxz/envs/sm_local/bin/python tools/export_student.py \
  --ckpt logs/loco/student_full_s0/ckpt_010000.pt \
  --out logs/loco/student_full_s0/export

/home/lxz/envs/sm_local/bin/python tools/monitor_loco.py \
  --root logs/loco --output logs/loco/monitor.json --poll_seconds 60 \
  --min_iterations 4000 --patience 3000 --level_delta 0.01 --success_delta 0.01 \
  --eval_max_lag 1000 --eval_map_noise on \
  --max_blowups 1000 --blowup_window 1000 --max_hours 24
```

The required local tests were run with these commands (training/lifecycle output directories must be fresh for a new run):

```bash
/home/lxz/envs/sm_local/bin/python tools/test_student.py --mode maps
/home/lxz/envs/sm_local/bin/python supermimic/train_student.py \
  --teacher logs/p2_full/ckpt_000200.pt --num_envs 1024 --iterations 100 \
  --log_every 50 --save_every 50 --log_dir logs/p3_student
/home/lxz/envs/sm_local/bin/python supermimic/eval_loco.py \
  --ckpt logs/p3_student/ckpt_000100.pt --map_noise both --trials 8
/home/lxz/envs/sm_local/bin/python tools/export_student.py \
  --ckpt logs/p3_student/ckpt_000100.pt --out logs/p3_student/export
/home/lxz/envs/sm_local/bin/python tools/test_student.py --mode monitor
/home/lxz/envs/sm_local/bin/python tools/test_student.py --mode lifecycle
/home/lxz/envs/sm_local/bin/python tools/test_loco.py \
  --mode semantics --out logs/p3_checks/p2_regression.json
/home/lxz/envs/sm_local/bin/python tools/export_student.py \
  --ckpt logs/p3_checks/finetune/last.pt --out logs/p3_checks/finetune/export
/home/lxz/envs/sm_local/bin/python supermimic/eval_loco.py \
  --ckpt logs/p3_checks/finetune/ckpt_000003.pt --map_noise both --trials 1
```
