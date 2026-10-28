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
- 2026-10-28 — **Eval bug found**: the legacy eval walked *backwards* out of the "down" pyramids, i.e. measured
  climbing stairs/slopes backwards; forward stair ascent was never measured (the 0% "stairs_down" scores were
  backward climbing). Added `eval_loco --direction forward` (`89f37a8`).
- 2026-10-28 — **Forward eval** (success over all levels / levels 7–9):

  | Terrain | Teacher full_s4 it 6000 | Student it 3000, map noise on | off |
  |---|---|---|---|
  | overall | 0.948 | 0.932 | 0.936 |
  | flat, rough, slopes up/down (≤25°) | 1.0 / 1.0 | 1.0 / 1.0 | 1.0 / 1.0 |
  | boxes (≤25 cm) | 1.0 / 1.0 | 0.95 / 0.83 | 0.95 / 0.83 |
  | stairs descend | 1.0 / 1.0 | 0.97 / 0.92 | 1.0 / 1.0 |
  | **stairs ascend** | **0.64 / 0.0** | **0.60 / 0.12** | 0.60 / 0.0 |

  Slopes, boxes, rough ground and stair descent are solved; the **student is insensitive to realistic map noise**
  and within ~0.02 of the teacher. Open gap: **ascending steps above ~15 cm** (levels 7–9, 17–20 cm).
- 2026-10-28 — P2.5 (stair ascent: foot clearance / stumble terms, outward-forward commands on stair tiles)
  delegated to Codex (first launch hung on stdin; relaunched with `< /dev/null`).
- 2026-10-28 — **Teachers converged** (monitor plateau stop at 11.5–14k iterations, mean level ~5.9). Longer
  training fixed much of the stair gap — forward success (all levels / levels 7–9):

  | Teacher | overall | stairs ascend | stairs descend | boxes |
  |---|---|---|---|---|
  | full_init4_s1 it 12651 | **0.982** | **0.88 / 0.67** | 1.0 / 1.0 | 1.0 / 1.0 |
  | full_s1 it 11985 | 0.979 | 0.90 / 0.67 | 0.95 / 0.83 | 1.0 / 1.0 |
  | full_s0 it 12052 | 0.977 | 0.85 / 0.54 | 1.0 / 1.0 | 0.99 / 0.96 |
  | full_s3 it 11557 | 0.975 | 0.89 / 0.62 | 0.94 / 0.79 | 1.0 / 1.0 |
  | full_s4 it 12392 | 0.971 | 0.80 / 0.42 | 1.0 / 1.0 | 1.0 / 1.0 |
  | full_s2 it 13964 | 0.911 | 0.70 / 0.46 | 0.90 / 0.75 | 1.0 / 1.0 |

  Student `student_s4` (distilled from the *early* it-3158 teacher) converged at 0.93 forward success, stair
  ascent 0.62 / 0.08 — it inherits its teacher's early weakness. New students are distilled from the final
  teachers (full_s3, full_s0, full_s1, and full_init4_s1 x {seed 42, seed 1, 10-frame history}); a PPO
  fine-tune of `student_s4` with the privileged critic runs alongside.
- 2026-10-27 — Started. Local test venv (`/home/lxz/envs/sm_local`, py3.13 + torch 2.9 + mujoco-warp 3.8) so Codex can test GPU code on the local RTX 4090.
