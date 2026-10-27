# P2 — privileged PPO terrain teacher

**The short-run learning criterion remains unmet:** both final 200-iteration runs improve reward and survival with zero NaNs/blow-ups, but mean terrain level stays zero. Evaluation has 0/560 successes and 12 falls. The implementation, collision stability, throughput, checkpoint lifecycle, and evaluation checks below are complete; longer-run locomotion convergence still needs validation.

Implementation and GPU validation on the local RTX 4090, using `/home/lxz/envs/sm_local/bin/python`. No network, new dependencies, or commits. `env.py`, `train.py`, and `evaluate.py` are unchanged. The pre-existing edits to `PERCEPTION_MILESTONES.md` and the task description are preserved.

## Implementation

- `robot.sim_model(..., collision="mesh" | "proxy")` retains the mesh default. The default compiled model is byte-identical to HEAD (75,078,727 bytes). Proxy mode preserves body masses, inertias, inertial frames, joint ranges, armatures, and actuator parameters exactly; visuals remain meshes.
- `LocoEnv` implements rsl-rl 5.2's `VecEnv` / `TensorDict` interface, 50 Hz control with four 5 ms implicitfast substeps, the existing PD/action scales and joint setpoint clamp, and one captured physics/contact graph per control step on `use_warp_stream`. Startup mass/CoM changes recompute MuJoCo-Warp constants, with writable derived model arrays expanded per world. Capacity is 128 contacts/world pooled across the batch, 32 CCD candidates/world pooled, and 1,000 constraint rows/world.
- Policy observations have 96 values: angular velocity, projected gravity, velocity command, relative joint positions, joint velocities, previous action. Uniform noise amplitudes are 0.2, 0.05, 0.01, and 1.5 for angular velocity, gravity, position, and joint velocity. The separate 187-point height scan is `base_z - terrain_z - standing_height`, clipped to [-1, 1] m. The actor consumes both groups (283 inputs); the critic receives their clean versions plus body-frame linear velocity and friction (287 inputs). `LocoCfg.scan_shape` and `scan_size` configure the grid.
- Commands are sampled every 8–12 s, with target vx/vy ranges [-1, 1] m/s and yaw rate [-0.5, 0.5] rad/s; XY ranges start at half scale, increase to 0.7 at 3,000 × 24 control steps and full scale at 8,000 × 24. There is a 20% forward-only share (positive vx, at least 0.3 m/s, zero vy/yaw), 10% standing, and 30% heading sampling (overridden by forward-only), heading gain 0.5. Resets use the owned tile center and random yaw. Curriculum promotes after >4 m displacement and demotes below half the *integrated commanded path length* over the episode; standing commands do not cause demotion. Solving the top row rerandomizes the level. Default initial level is zero, configurable with `--max_init_level`.
- Domain randomization: friction U[0.8, 1.2], torso mass/inertia scale U[0.8, 1.2], torso CoM offsets ±(0.025, 0.025, 0.03) m, additive horizontal velocity pushes ±0.5 m/s every 5–15 s, and observation noise. Randomization is disabled for evaluation.
- Falls terminate at relative base height <0.35 m or tilt >70 degrees. The 20 s timeout is bootstrapped using the **terminal observation before reset**, rather than the reset state or the previous state's value. Numerical guard thresholds match `env.py`: root speed component >50 or joint speed component >100, or nonfinite state/action. Invalid worlds roll back, reset solver state, and get zero reward. `ValidPPO` cuts their GAE continuation and omits their samples from all PPO losses/minibatches.
- `LocoRunner` subclasses `OnPolicyRunner`, using its model/algorithm construction and stock PPO optimization. Its small loop writes CSV every 50 iterations and atomic `ckpt_NNNNNN.pt` / `last.pt`. STOP is checked each iteration; resume restores actor, critic, optimizer, normalizers, adaptive LR, terrain levels, counters (including command-curriculum time), and RNG state, starting fresh episodes. `--iterations` is the total target iteration. Logging is local CSV; no external logger is started.
- `eval_loco.py` uses deterministic actor means, all seven terrain types and every difficulty row, 8 s at ±0.6 m/s (negative for names ending in down), and success = displacement >3 m with no fall during the entire trial. It writes `eval_loco.csv` and `*_eval.json` even with `--ckpt`, and supports a watcher. Default eight trials use deterministic yaw offsets within ±5 degrees, with exact tile-center positions. Tracking error averages only the portion before the first fall. P1 terrain names describe pyramid sign: from the center, a positive pyramid slopes downward outward; the evaluator preserves the task's requested naming/command convention.

## Reference choices

Local reference root: `/home/lxz/mjlab-loco/src/mjlab/tasks/velocity/`.

| Reference file / lines | Adopted choice or explicit adaptation |
|---|---|
| `config/g1/rl_cfg.py:72–114` | Actor/critic ELU MLPs 512/256/128, empirical normalization, 24 steps/env, 5 epochs, 4 minibatches, adaptive LR 1e-3, target KL .01, entropy .01, clipping .2, gamma .99, lambda .95, gradient limit 1.0. Initial action std is reduced from 1.0 to 0.5 for this actuator/contact setup; advantages normalize per minibatch after invalid-sample exclusion. |
| `velocity_env_cfg.py:74–103, 173–190` | Proprioceptive noise, velocity ranges, 10% standing, 30% heading, heading gain .5; 20% forward-only follows `mdp/velocity_command.py:135–136, 175–199`. Command period changed to the requested ~10 s; height scan stays privileged/noiseless. |
| `velocity_env_cfg.py:234–266` | Friction range and torso CoM ranges. Added mass/inertia scaling; pushes are gentler and less frequent than the reference's 1–3 s full-twist perturbations. |
| `velocity_env_cfg.py:271–380`; `mdp/rewards.py:42–76, 699–759, 927` | Linear/yaw exponential tracking weights 2/2 and variance .25; upright exponential weight 1, variance .2; terrain-relative height exponential weight 1, variance .01; joint-limit penalty -1 and foot-slip penalty -.1. Fixed standing-height target replaces the reference's additional height command. |
| `config/g1/env_cfgs.py:403–424, 1381–1383, 1445` | Roll/pitch rate -.05; termination -10 and torque squared -1e-5 from the G1 MoRE rough variant. Action-rate weight -.03 also follows that variant (`:1384`), reducing the base config's -.1 to allow stepping. |
| `mdp/rewards.py:320–351`; `config/g1/env_cfgs.py:406–407` | Feet-air-time intent and command gating. Enabled a .4-weight biped single-support duration reward (capped .5 s); the current base G1 config disables its air-time reward. Both-feet flight gets no reward. |
| `velocity_env_cfg.py:384–396`; `mdp/curriculums.py:46–115` | 70-degree fall tilt, timeout bootstrap, displacement curriculum, per-type logging. Integrated commanded distance handles resampling within an episode; initial level zero makes promotions observable. The command-range warm-up uses the schedule in `velocity_env_cfg.py:405–428`, scaling XY ranges .5 → .7 → 1.0; unlike that reference, final vy also reaches ±1 and yaw remains ±.5. |
| `config/g1/env_cfgs.py:270–276, 486–493` | Investigated rough CCD budget 500; use the terrain-scan variant's 50. More iterations did not fix the captured foot-pad failure. CCD tolerance is 1e-4 m for the large FP32 terrain. |

Additional requested regularizers: vertical velocity -2; joint acceleration squared -2.5e-7; non-foot normal-force contacts over 10 N, -1 per body; hip roll/yaw, arms, and waist L1 deviation each -.1. All rates, including termination, are multiplied by control dt = .02. Rewards are not clipped to positive. We do not reproduce the reference's full variable-posture distribution, angular-momentum sensor, learned motion priors, or terrain-specific foothold rewards; this implementation uses the requested explicit deviation penalties and a biped support reward. Self-collisions remain disabled as in the original repository.

## Contact extraction and proxy geometry

Within the single control-step CUDA graph, after **each** `mjw.step`, a Warp kernel traverses valid `data.contact` entries (`i < nacon`), reads their `worldid`, `geom`, and four `efc_address` rows, and sums `data.efc.force` for the pyramidal condim-3 contact. Those four nonnegative pyramid forces sum to the contact's normal load. Loads are accumulated into both participating non-world bodies via `model.geom_bodyid`, averaged over the four substeps. This is scalar summed normal load, **not** the norm of a net vector force. Feet use >1 N; undesired bodies use >10 N. Foot slip uses body spatial velocity corrected from root-subtree CoM to foot origin. The final kinematics refresh avoids stale link poses. No contact tensor is copied to CPU in the environment.

Meshes are fitted in their compiled principal frames (limbs) or link frames (boxes), accounting for MuJoCo's mesh recentering/rotation. Long capsules split into overlapping segments with axis length ≤10 cm; boxes split into cells with sides ≤12 cm. The 50-contact/pair cap applies to primitives too, so replacing a large mesh with one large box alone is insufficient.

Sizes below are meters: box **half-extents**, capsule **radius / half-axis-length**, cylinder **radius / half-length**. Left/right pairs have effectively identical sizes; counts are **per side** for limb rows. Full precision and both sides are in `logs/p2_checks/models.json`.

| Body | Primitive count | Size |
|---|---:|---|
| pelvis contour | 4 boxes | .059780, .033884, .031060 |
| hip pitch | 1 capsule | .050477 / .015685 |
| hip roll | 1 capsule | .053941 / .036495 |
| hip yaw / thigh | 2 capsules | .057979 / .037600 |
| knee / shin | 3 capsules | .053571 / .041515 |
| ankle pitch | 1 capsule | .017084 / .006116 |
| foot | 4 spheres | radius .025; y centers ±.02; original sole bottom preserved |
| torso | 12 boxes | .037567, .053857, .053421 |
| logo | 1 capsule | .076083 / .001000 |
| head | 8 boxes | .035085, .038898, .051444 |
| shoulder pitch | original cylinder | .030000 / .025000 |
| shoulder roll | original cylinder | .030000 / .015000 |
| upper arm | 1 capsule | .042971 / .035031 |
| elbow / forearm | 1 capsule | .037461 / .032341 |
| wrist roll | 1 capsule | .030001 / .001000 |
| wrist pitch | 1 capsule | .032858 / .019013 |
| wrist yaw | 1 capsule | .030911 / .001000 |

A failed early prototype used 15 mm foot pads and the default 20 ms contact time constant. CPU/GPU replay isolated a bogus GPU contact distance of **-4.364963 m** while CPU penetration was **-0.016203 m**, with the pad center inside a hfield prism (`logs/p2_checks/replay.log`). Increasing CCD iterations or disabling warm start did not cure it. Final 25 mm pads and a 10 ms contact time constant completed **1,024 worlds × 1,000 random-action steps with zero blow-ups** (`pads_random.log`). This was the substantive stability fix; discarded exploratory training results are not counted as passing validation.

## Final GPU results

The required two runs used 1,024 environments, seed 42, 200 iterations each (4,915,200 transitions/run). Both completed with **zero blow-ups**, finite logged losses/rewards, and rising reward and episode duration. **Neither reached a curriculum promotion within 200 iterations.** Consequently the requested short-run terrain-level improvement is not demonstrated by these two runs; it is not inferred from synthetic curriculum tests.

| Terrain | Iteration | Mean step reward | Episode return | Episode length (steps) | Mean level | XY tracking error (m/s) | Blow-ups |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 50 | 0.03121 | 3.20 | 92.7 | 0.0000 | 0.669 | 0 |
| flat | 100 | 0.03969 | 7.58 | 181.3 | 0.0000 | 0.563 | 0 |
| flat | 150 | 0.04610 | 22.98 | 494.6 | 0.0000 | 0.439 | 0 |
| flat | 200 | 0.04695 | 30.37 | 645.2 | 0.0000 | 0.414 | 0 |
| full | 50 | 0.03129 | 3.24 | 94.0 | 0.0000 | 0.671 | 0 |
| full | 100 | 0.03932 | 7.52 | 178.7 | 0.0000 | 0.567 | 0 |
| full | 150 | 0.04589 | 23.79 | 515.6 | 0.0000 | 0.442 | 0 |
| full | 200 | 0.04693 | 30.23 | 639.6 | 0.0000 | 0.415 | 0 |

Artifacts: `logs/p2_flat/train_log.csv`, `logs/p2_full/train_log.csv`, both `ckpt_000200.pt` checkpoints, and `logs/p2_checks/{flat,full}_curriculum_stdout.log`. Flat training took 99.46 s for the learning loop; full training took 132.46 s (initial model setup/compilation excluded). Per-type terrain levels are separate CSV columns, initially all zero. An earlier uniform-command version was also continued from 200 to 1,000 iterations: zero blow-ups, but level remained zero and reward plateaued (0.0331 at 1,000). This motivated adopting the reference command warm-up and forward-only share before the final two runs. Those exploratory artifacts are in `logs/p2_full_uniform_commands/` and `logs/p2_checks/full_extension_stdout.log`; they are not results for the final command configuration.

| Other GPU check | Result |
|---|---|
| P1 standing/penetration, proxies, seven types × rows 0/5/9 | PASS: 21 worlds, 200 substeps, 19,744 surface probes/robot; worst penetration 0.002296 m; peak total contacts 359, peak constraints/world 72 |
| GPU sampler vs CPU terrain rays | PASS, max error 0.00873953 m; original P1 exact convention and yaw-scan checks retained |
| Contact extraction sanity | Mean foot normal load 322.472 N vs robot weight 327.077 N; non-foot load zero in standing |
| Curriculum branches | PASS: promotion, demotion, standing-command neutrality, top-row rerandomization, exact tile-center reset |
| Timeout / invalid transition handling | PASS: pre-reset bootstrap value checked numerically; injected NaN action rolls back/resets; 191 of 192 synthetic transitions enter PPO batches; excluded GAE advantage is zero |
| Command curriculum / scan configuration | PASS: XY bounds at all three command stages; 5 × 3 scan yields 15 height values / 115 critic values; single-world logs stay finite |
| STOP / resume / watch | PASS: STOP saves iteration 200 without changing actor weights; resume advances to 201 and changes weights, preserves command-curriculum clock; watcher writes exactly 70 unique terrain/level rows (then intentionally stopped with `timeout`) |
| Regression | Default mesh compiled model byte-identical to HEAD; proxy masses/inertias/actuators unchanged |
| Random-action stress | 1,024 × 1,000 steps, N(0, .5²) actions, zero blow-ups |

Full-environment benchmark, **4,096 worlds**, full P1 terrain, 3 × 50 control steps after warm-up; includes physics, per-substep contacts, rewards, observations/noise/187-point scan, logging metrics, commands, termination/reset checks. Fixed default targets; initial level zero; no domain randomization for a matched geometry comparison. Synchronization brackets every timed trial. These are 50 Hz **world control steps/s**, not physics substeps/s.

| Collision | Three trials (steps/s) | Median | Blow-ups |
|---|---|---:|---:|
| mesh | 103,415 / 105,042 / 104,290 | **104,290** | 0 |
| proxy | 125,255 / 126,851 / 127,533 | **126,851** | 0 |

Proxy speedup: **1.22×**. The independent P1 simulation-only proxy benchmark gave 155,240 steps/s on terrain and 469,631 on a plane; its original 35-iteration CCD/600-constraint settings differ from LocoEnv, so use the full-environment comparison for training estimates. Benchmark JSON/logs and final penetration log are under `logs/p2_checks/`.

`eval_loco` ran the full-terrain 200-iteration checkpoint on **560 trials** (7 types × 10 levels × 8 headings): **0% success, 12 falls, 0 blow-ups**. The policy mostly stands: mean maximum displacement by type is 0.013–0.160 m, well below 3 m. Per-type mean XY tracking errors across levels:

| flat | rough | slope up | slope down | stairs up | stairs down | boxes |
|---:|---:|---:|---:|---:|---:|---:|
| 0.6012 | 0.6012 | 0.6006 | 0.5956 | 0.6000 | 0.5958 | 0.6364 |

Both final checkpoints and every CSV numeric field were explicitly checked finite; no final training/contact/benchmark log contains an overflow or traceback. AST parsing and `git diff --check` pass. Integration evidence is in `logs/p2_checks/integration.json`.

Detailed per-type/per-level results: `logs/p2_full/eval_loco.csv` and `logs/p2_full/ckpt_000200_eval.json`. Stable standing on the spawn platform is **not** evidence of terrain traversal. The 200-iteration checkpoint is a smoke-test checkpoint, not a qualified locomotion teacher.

Reproduction commands (all from the repository root):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python supermimic/train_ppo.py --num_envs 1024 --iterations 200 --terrain flat --save_every 200 --log_dir logs/p2_flat
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python supermimic/train_ppo.py --num_envs 1024 --iterations 200 --terrain full --save_every 200 --log_dir logs/p2_full
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python supermimic/eval_loco.py --ckpt logs/p2_full/ckpt_000200.pt
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python tools/test_loco.py --mode models --out logs/p2_checks/models.json
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python tools/test_loco.py --mode semantics --out logs/p2_checks/semantics_final.json
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python tools/test_loco.py --mode benchmark --collision mesh --out logs/p2_checks/benchmark_mesh.json
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python tools/test_loco.py --mode benchmark --collision proxy --out logs/p2_checks/benchmark_proxy.json
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python tools/test_terrain.py --collision proxy --preview logs/p2_checks/terrain_proxy.png
```


## Launch

From `/home/lxz/supermimic_vibed`, one process per visible GPU:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python -u supermimic/train_ppo.py \
  --num_envs 4096 --iterations 30000 --seed 42 --collision proxy \
  --terrain full --max_init_level 0 --save_every 1000 --log_dir logs/loco_teacher
```

Resume by adding `--resume logs/loco_teacher/last.pt` and retaining the desired total `--iterations`. Create `logs/loco_teacher/STOP` to save and exit; remove that file before resuming. Evaluation watcher:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python -u supermimic/eval_loco.py \
  --run_dir logs/loco_teacher --watch
```
