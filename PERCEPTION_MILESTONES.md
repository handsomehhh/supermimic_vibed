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
- 2026-10-28 — Best teacher: **full_init4_s0 it 16717, forward success 0.984**. Students at ~3300 iterations:
  legacy success 0.90–0.94; **PPO fine-tune of the student lifts it from 0.87 to 0.94** (`student_s4_ppo`); a
  10-frame history student (0.934) beats its 5-frame twin (0.911).
- 2026-10-28 — **P2.5 merged** (`6c77244`, Codex + review): opt-in `--preset stairs_v1` = foot clearance (-0.25,
  10 cm target, from mjlab MoRE rough) + foot stumble (-1.0) + 50% outward-forward commands on pyramid tiles;
  `diagnose_stairs.py`. Paired fine-tuning experiment (+5000 iterations each from the best teachers):
  `ft_i4s0_stairs` (preset), `ft_i4s0_control` (continued training, no preset), `ft_i4s0_outward` (outward
  commands only), `ft_i4s1_stairs` (preset on the second teacher). Freed GPUs by stopping 4 redundant students.
- 2026-10-28 — Infra fixes during P2.5 (each cost ≤ 30 min of GPU time):
  - monitor counted the *cumulative* blow-up counter restored from a checkpoint as recent → stopped resumed runs
    at once (`dca6eff`); then counted rewritten iterations after a resume-from-older-checkpoint twice
    (`6ec9bf8`: dedupe by iteration, a decrease is a restart);
  - `HeightSampler.sample` turned NaN query points (foot positions of a blown-up world, read before the rollback)
    into out-of-range indices → device-side assert in the stairs preset after ~1500 iterations (`3ebd56e`).
- 2026-10-28 — `student_full_s1` converged (legacy success 0.934 at 5.7k iterations) → PPO fine-tune started.
- 2026-10-28 — **Evaluation noise**: with one fixed tile and near-identical ±5° headings per cell (deterministic
  rollouts), per-cell success swung by ±0.2 between eval variants (±5° vs ±30° even reversed the ranking). Added
  `eval_loco --robust` (random ±30° headings, ±0.25 m start offsets, every tile of the type; 32 trials per cell,
  96 per hard cell) and report Wilson 95% intervals.
- 2026-10-28 — **P2.5 result (robust eval, forward, ±30°)** — stair ascent at levels 7–9 (17–20 cm):

  | Teacher | overall | ascend L7–9 (95% CI) | descend L7–9 | boxes L7–9 |
  |---|---|---|---|---|
  | source full_init4_s0 it 16717 | 0.992 | 0.917 (0.84–0.96) | 0.927 | 1.000 |
  | control: +3400 it, no change | 0.984 | **0.750 (0.65–0.83)** | 0.990 | 0.948 |
  | `stairs_v1` preset | 0.998 | 0.958 (0.90–0.98) | 1.000 | 1.000 |
  | **outward-forward commands only** | **1.000** | **0.990 (0.94–1.00)** | 1.000 | 1.000 |
  | second teacher + `stairs_v1` | 1.000 | 0.990 (0.94–1.00) | 1.000 | 1.000 |

  **Command shaping is what fixes stair ascent**; foot clearance / stumble rewards add nothing on top. Continued
  training with the original command mix *degrades* ascent (0.92 → 0.75): sideways/backward/turning commands
  dominate and forward climbing is rarely practiced. New best teacher: `ft_i4s0_outward`. Control stopped;
  10-frame-history student distillation from it started.
- 2026-10-28 — **First deployable artifact (v0)**: `student_s4_ppo` it 8223 (distilled from the early it-3158
  teacher, then PPO fine-tuned with the privileged critic), exported to `logs/loco/student_s4_ppo/export/`
  (`student.ts`, `student.onnx`, `student_spec.json`). Robust eval (forward, ±30°, 32 trials, noisy map):
  **0.960** overall; flat/rough/slopes/stair descent 1.00 at every level, boxes 0.99 (0.96 at L7–9),
  **stair ascent 0.73 (0.17 at L7–9)** — inherits the early teacher's weakness; clean vs noisy map within 0.01.
  Students from the outward-command teachers (`student_outward_h10`, `student_i4s1stairs_h10`) should close it;
  a from-scratch teacher with outward commands (`scratch_outward_s0`) checks the fix without fine-tuning.
- 2026-10-28 — **Deployable v1**: `student_outward_h10_ppo` it 3000 (10-frame history student distilled from the
  outward teacher, then PPO fine-tuned), exported to `logs/loco/student_outward_h10_ppo/export_v1/`. Robust eval,
  noisy map: **0.983** overall; stair ascent L7–9 **0.67** (v0: 0.17), descent L7–9 0.98, boxes L7–9 0.99,
  slopes 1.0. Other students from the new teachers plateau at robust 0.94–0.98, ascent L7–9 0.43–0.65.
- 2026-10-28 — **Diagnosis: perception noise is not the bottleneck.** The same v1 student with an exact, fully
  valid map (no noise, no blind zone, no latency): overall 0.979, ascent L7–9 **0.58** (vs 0.67 noisy). The
  teacher–student gap on 17–20 cm steps (0.99 vs 0.67) comes from the student policy / distillation, not the
  map. Next: larger map encoder, stair-weighted distillation or longer PPO on stairs. Running for reference: a
  clean-map student (upper bound) and a PPO replicate (seed 1). From-scratch outward teacher at 6000 iterations:
  robust 0.983, ascent L7–9 0.885 (still improving) — confirms the command fix without fine-tuning.
- 2026-10-27 — Started. Local test venv (`/home/lxz/envs/sm_local`, py3.13 + torch 2.9 + mujoco-warp 3.8) so Codex can test GPU code on the local RTX 4090.
- 2026-10-28 — **Student capacity is the stair bottleneck (P4).** Distillation from the same outward teacher, robust eval,
  stair ascent L7–9 (overall in brackets): CNN map encoder + MLP [512,256,128] on stair-weighted terrain (30% ascent
  tiles) **0.82** at it 4000 (0.980); CNN on default terrain 0.78 at it 5000 (0.984); wide MLP only 0.68 peak, noisy
  (0.48–0.68); 20-frame history no better (0.22–0.58); clean-map student 0.58 (confirms noise is not the limit).
  Baseline MLP [256,128] students: 0.43–0.65. The spatial map encoder matters more than width or history.
  PPO fine-tune of the v1 recipe, seed 1: 0.77 at it 7000 (overall 0.988). From-scratch outward teacher
  (`scratch_outward_s0`) reaches 0.999 overall / 0.99 ascent at it 12000, matching the fine-tuned teacher.
  Next: PPO fine-tune of the CNN-stairs student (stair-weighted and default terrain), seed replicate, and
  distillation from the from-scratch teacher.
- 2026-10-28 — **v2 candidate, not yet a v1 replacement.** `student_cnn_h10_stairs` it 4000 exported to
  `logs/loco/student_cnn_h10_stairs/export_v2/`. Robust L7–9, noisy map: ascent **0.82** (v1 0.67), but descent
  0.81 (v1 0.98) and boxes 0.94 (v1 0.99); overall 0.980 vs 0.983. Noise off: ascent 0.83, so noise is still not
  the limit. Stair-weighted distillation shifts skill toward ascent at the cost of descent. The PPO fine-tune of
  this student does the reverse at it 1000 (ascent 0.49–0.55, descent 0.96–0.99, boxes 0.97): early PPO trades
  ascent away even with outward commands on. Single robust evals vary by about ±0.07 per bucket (same weights:
  0.82 and 0.90 in two evals), so decisions need several checkpoints. Queue runner (`tools/gpu_queue.sh`) now
  refills idle GPUs automatically.
- 2026-10-28 — **Deployable v2: the teacher was the missing piece.** Same CNN-stairs student recipe (CNN map encoder,
  MLP [512,256,128], 10-frame history, 30% ascent tiles), distilled from the from-scratch outward teacher
  (`scratch_outward_s0` it 12000) instead of the fine-tuned one: `student_cnn_stairs_scratchT` it 2000 exported to
  `logs/loco/student_cnn_stairs_scratchT/export_v2/`. Robust L7–9, noisy map: overall **0.996**, ascent **0.96**,
  descent 0.99, boxes 0.96 (it 1000: 0.991 / ascent 0.97). Noise off: 0.997, ascent 0.94, descent 1.0, boxes 0.99.
  v1 was 0.983 / ascent 0.67. The fine-tuned teacher (stair fine-tune on top of a flat/rough teacher) scores the
  same as this teacher itself, but its behaviour distils worse; a teacher trained with outward commands from the
  start gives a student that keeps both ascent and descent. Replicates queued (seed 1; second from-scratch
  teacher). Also seen: a student distilled on a clean map collapses on the noisy map (0.0), and a 20-frame
  history is unstable (0.83–0.97).
- 2026-10-28 — **Depth camera works as well as the elevation map.** Student fed a single forward depth image
  (32x24, GPU ray-marched from the terrain, D435-like noise; `--observation depth`, CNN encoder, MLP
  [512,256,128], 10-frame history), distilled from the from-scratch teacher (`scratch_outward_s0` it 12000).
  Robust L7–9: default terrain mix (`student_depth_scratchT_mix`) it 3000 overall **0.996**, ascent **0.95**,
  descent 0.99; stair-weighted mix it 2000 0.995 / 0.94 / 0.95 (later checkpoints drift to ascent 0.86). So no
  elevation-mapping stack is needed for these terrains. Queued: seed 1, PPO fine-tune from it 3000, noise-free
  depth (upper bound).
- 2026-10-28 — **v2 recipe replicates.** Map student, seed 1 (`student_cnn_stairs_scratchT_s1`) it 2000: 0.998 /
  ascent 0.98. From the second from-scratch teacher (`scratch_outward_s1` it 7112): 0.994 / 0.91. Without the CNN
  (MLP map encoder, same teacher, stairs mix): 0.995 / 0.93 at it 5000 — with a good teacher the CNN matters less.
  5-frame history: 0.991 / 0.96 at it 4000, unstable after. `tools/play_loco.py` plays any of these locally.
