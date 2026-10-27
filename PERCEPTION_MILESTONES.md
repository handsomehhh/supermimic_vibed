# Perceptive velocity locomotion — milestones

Goal: a deployable velocity-command policy for the G1 that perceives terrain (elevation map; depth later)
and crosses simple terrain: stairs (up/down), slopes, platforms, rough ground. Training on the 8-H100
cluster under the same rules as exp1–4 (all GPUs busy, stop rules + monitor, hourly check-in).
Coding tasks are delegated to local Codex (gpt-6-astra, xhigh); Claude plans, reviews and runs experiments.

## Plan

| Phase | What | Exit criterion |
|---|---|---|
| P1 | Terrain in MuJoCo-Warp: one shared heightfield of terrain tiles (flat, rough, slopes, stairs, platforms) x difficulty levels; contact check vs CPU MuJoCo; GPU height sampling (scandots, elevation map) | robot stands/walks on every tile type without penetration; sampler matches the heightfield |
| P2 | Teacher: PPO velocity tracking with privileged height scan + terrain curriculum | reaches the hardest level on stairs/slopes/platforms in most envs |
| P3 | Deployable student: noisy robot-centric elevation map (drift, holes, latency) + proprio history, asymmetric critic or DAgger from P2 | close to the teacher on all terrains with realistic map noise |
| P4 | SuperMimic-style: terrain-aware world model + velocity encoder with elevation map (BPTT), vs P2/P3 | — |

## Log

- 2026-10-28 — **P1 done** (`ec0012b`, Codex + review). One shared 90 x 170 m hfield, 7 tile types
  (flat, rough, slope up/down, pyramid stairs up/down, boxes) x 10 difficulty levels (stairs up to 0.20 m,
  slopes ~25°, boxes 0.25 m). GPU height sampler matches MuJoCo rays to 8.7 mm (max; p99 1 mm); the robot
  stands on every tile type with < 3 mm penetration. Throughput at 4096 worlds: 145k steps/s on terrain vs
  521k on the plane (hfield–mesh collisions dominate). Risk found: MuJoCo-Warp caps hfield–convex-mesh contacts
  at 50 per pair; an open-loop fall on rough ground overflowed it and blew up (120 rad/s) → P2 adds primitive
  collision proxies for terrain training.
- 2026-10-28 — **P2 code done** (`ba44978`, Codex + review): `LocoEnv` (rsl_rl VecEnv), PPO teacher with privileged
  187-point height scan, rewards/curriculum/commands adapted from the user's mjlab G1 rough config, per-body
  contact forces from MuJoCo-Warp contacts (standing foot load 322 N vs weight 327 N), collision proxies
  (capsules/boxes, 25 mm foot spheres, 10 ms contact time constant — fixed a GPU-only bogus −4.36 m hfield contact
  on 15 mm pads), STOP/resume, `eval_loco` (success = 3 m traversal per type x level). Local 200-iteration smoke
  runs: reward and episode length rise, no blow-ups, no curriculum promotion yet (too short).
- 2026-10-28 — **P2 batch 1 on the cluster** (8 x H100, 4096 envs each, ~0.56 it/s): full terrain seeds 0–4, flat
  seed 0, full with initial level up to 4 (seeds 0, 1). Finding: **mesh collisions are unusable on terrain** —
  111k blow-ups within 50 iterations, reward ~0; proxies: 0 blow-ups. Mesh run replaced by seed 4.
- 2026-10-28 — **P3 code done** (`381cc4c`, Codex + review). Elevation map (2.0 x 1.2 m at 8 cm, terrain height
  relative to the acquisition base + validity mask) with per-cell noise 1–3 cm, ±5 cm drift, ±5 cm / ±3°
  registration error, 0–100 ms latency, ~10 Hz refresh, 5–10% dropout, body blind zone, sensor shadowing.
  Student = 5-frame proprio history + map (1230 inputs, no privileged data); DAgger distillation (rsl_rl
  `DistillationRunner`) then optional asymmetric PPO fine-tune; TorchScript/ONNX export + input spec (max
  deviation 6e-7); CPU monitor (`tools/monitor_loco.py`, running on the cluster in tmux `loco_monitor`).
- 2026-10-28 — Teachers at ~700 iterations: first curriculum promotions (mean level 0.02–0.15).
- 2026-10-28 — **Teachers walk on terrain by ~3000 iterations** (~1.5 h at ~1 it/s): mean terrain level 3.9–5.1 of
  9, eval success over all 7 types x 10 levels **0.80–0.90**, tracking error 0.17–0.20 m/s. Flat-only control:
  level n/a, plateau at 13k iterations.
- 2026-10-28 — **Mistake: monitor blow-up threshold was absolute (1000 per 1000 iterations)** — with 4096 envs x 24
  steps that is 0.001% of transitions, which the guard handles harmlessly; it stopped all 7 healthy teachers at
  ~3000 iterations and the GPUs sat idle ~3 h before it was noticed. Fixed: threshold 50k per 1000 iterations
  (~0.05%), teachers resumed from `last.pt`, plus an idle-GPU/STOP alert watcher.
- 2026-10-28 — **Student distillation started** on GPU 4 from teacher `full_s4` (it 3158, success 0.90;
  snapshot `logs/loco/teacher_full_s4_it3158.pt`), noisy elevation map: after 150 iterations mean level 4.8,
  tracking error 0.16 — imitates the teacher quickly.
- 2026-10-27 — Started. Local test venv (`/home/lxz/envs/sm_local`, py3.13 + torch 2.9 + mujoco-warp 3.8) so Codex can test GPU code on the local RTX 4090.
