# P2.5 — opt-in stair-ascent fine-tuning

The implementation and local stability checks are complete. **Improved level-7–9 teacher success is not established locally.** The available P2 iteration-200 checkpoint cannot walk; it was used only for loading/resume validation. The updated baseline from the user is 0.80–0.90 overall forward ascent success and 0.42–0.67 at levels 7–9. The experiment to run is fine-tuning those final walking teachers, with matched forward evaluation before and after.

All executed Python used `/home/lxz/envs/sm_local/bin/python`, on the local RTX 4090. No network, new dependencies, commits, or edits outside this repository. Pre-change revision: `9df2460c8496602d52fa1d7ed0473f79a1a7183a`.

## Changes and defaults

`supermimic/loco_env.py`, `supermimic/stairs.py`, and `supermimic/train_ppo.py` add the following options. Explicit CLI values override the preset, which overrides the loaded checkpoint configuration. Omitted flags preserve a resumed checkpoint's settings.

| LocoCfg field / CLI flag | Default | `--preset stairs_v1` |
|---|---:|---:|
| `foot_clearance_weight` | 0 (off) | -0.25 |
| `foot_clearance_target` | 0.10 m | 0.10 m |
| `foot_stumble_weight` | 0 (off) | -1.0 |
| `foot_stumble_ratio` | 3.0 | 3.0 |
| `outward_fraction` | 0 (off) | 0.5 |
| `outward_speed` | 0.6 m/s | 0.6 m/s |
| `contact_vectors` | false | enabled internally by stumble; explicit flag is for diagnostics |

The clearance cost is `sum(abs(clearance - target) * horizontal_foot_speed * swing)`, active when commanded XY speed plus absolute yaw rate exceeds 0.05. `HeightSampler` samples underneath every spherical sole pad, including the toe. Clearance is the minimum pad-bottom-to-terrain gap on each foot, not ankle height or height above the tile origin. Swing means the existing scalar foot contact load is at most 1 N. As with existing rewards, weights are multiplied by the 0.02 s control timestep. This adds swing gating and conservative sole-pad sampling to the reference term.

Stumble is the reference binary cost: any foot with `norm(Fxy) > 3 * abs(Fz)`. The optional accumulator decodes all four pyramidal constraint forces, applies friction, rotates the contact-frame force into world coordinates, and applies the force to the correct side of each contact. Vectors are summed per body and averaged over all four physics substeps. Existing scalar normal-load accumulation is unchanged. Opposing contacts can cancel in this net-vector signal, and a 20 ms average is not a peak impact force.

Both knee/shin bodies are **already included** in `body_contact`: -1 per non-foot body above 10 N. No duplicate penalty was added. The diagnostic records each knee/shin body's scalar normal load and vector force independently.

Outward commands apply to both signs of stairs and slopes. The configured fraction of eligible environments receives positive body-frame forward speed, zero lateral speed, and a heading toward the nearest rectangular tile edge. Ties favor the current heading. At reset, selected robots face that edge. On later command resampling, yaw turns toward the selected edge and forward speed is multiplied by `max(cos(heading_error), 0)`. Selection overrides standing for those environments. Speed follows the existing command warm-up: 0.3 → 0.42 → 0.6 m/s with the preset. The other environments retain the original command sampler.

The default branches allocate no new contact-vector/probe buffers, launch no new physics kernels, add no reward terms, and consume no extra random numbers. Observation groups and network sizes remain 96/187/287. `cfg_to_dict` omits P2.5 fields equal to their defaults, preserving the old serialized environment configuration. Fine-tuning restores actor, critic, normalizers, optimizer, learning rate, curriculum levels and counters, then starts fresh episodes as before. `--iterations` is still the **total target iteration**, not the number of additional iterations.

CSV headers are migrated atomically if a resumed run enables or disables reward columns. Historical rows retain their data; unavailable historical terms are blank. A new log directory is preferable when comparing fine-tuning against the source teacher.

## Local mjlab references

Read-only reference root: `/home/lxz/mjlab-loco/src/mjlab/tasks/velocity/`.

| Source | Adopted choice |
|---|---|
| [`config/g1/env_cfgs.py:1429`](../../mjlab-loco/src/mjlab/tasks/velocity/config/g1/env_cfgs.py) (lines 1429–1443) | G1 MoRE rough variant: clearance weight -0.25, target 0.10 m, command threshold 0.05; stumble weight -1.0. |
| [`mdp/rewards.py:371`](../../mjlab-loco/src/mjlab/tasks/velocity/mdp/rewards.py) (lines 371–398) | Absolute terrain-relative foot-clearance error weighted by horizontal foot speed; linear plus angular command gating. |
| [`mdp/MoRE/rewards.py:496`](../../mjlab-loco/src/mjlab/tasks/velocity/mdp/MoRE/rewards.py) (lines 496–514) | Any-foot horizontal/vertical force ratio, default threshold 3.0. |
| [`mdp/rewards.py:1234`](../../mjlab-loco/src/mjlab/tasks/velocity/mdp/rewards.py) (lines 1234–1249) | Equivalent documented rough-terrain stumble term. |

The 50% outward share and heading/translation coupling are new task-specific choices, not copied mjlab settings. The more aggressive clearance variants elsewhere in `env_cfgs.py` were not used as the first fine-tuning preset.

## Diagnostics

New tool: `supermimic/diagnose_stairs.py`. It defaults to **forward** commands on `stairs_down`, levels 7/8/9, 8 trials per level, 8 seconds at 0.6 m/s, and the evaluator's deterministic ±5° starting headings. It supports teacher and student checkpoints; student map noise can be switched off with `--map_noise off`. It does not change `eval_loco` or its legacy default direction.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python supermimic/diagnose_stairs.py \
  --ckpt logs/loco/full_s4/last.pt --levels 7 8 9 --trials 16 \
  --seconds 8 --speed 0.6 --plots 6 --out logs/p25_diagnostics/full_s4_before
```

Outputs:

- `trajectory.npz`: time, level, tile origin, root qpos/qvel, actions, all sole-pad world positions, terrain under each pad, minimum foot clearance, foot/shin vector forces and scalar normal loads, base height, tilt, validity and first-fall masks, and a sampled stair profile.
- `summary.json`: per-trial outcome, maximum displacement, end position, actual termination thresholds, near-end contact hypotheses, and peak forces. `stall` means no traversal and less than 5 cm net motion over the last second; other unfinished trials are `incomplete`.
- Up to `--plots` PNGs: sole trajectories and base path over the stair profile, foot/shin forces over time, and clearance over time. Plots cover each requested level before adding more trials from one level, prioritizing failures within each level.

The `alive` mask includes the first fall; all later reset states are excluded. `valid` excludes numerical rollback states. Force flags are correlations, **not a claim that a riser catch caused a particular teacher failure**. The archive preserves the raw evidence for that assessment.

Local synthetic recorder check starts the default-pose robot 0.60 m outward with initial forward velocity 0.8 m/s and holds its default joint targets. This intentionally creates failures near the first riser without pretending the local P2 checkpoint is a walking teacher. The recorded failures have knee/shin contact and little sole lift. See the validation results below.

## Curriculum check

The existing curriculum remains unchanged. Levels are stored per environment; terrain columns/types remain fixed, and separate type averages are already logged. Promotion requires final displacement greater than 4 m on the default 8 m tile. Demotion requires displacement below half the integrated commanded XY path, unless promoted. Standing has no travel requirement. This is a displacement curriculum, not a count of climbed steps or a fall-aware completion test.

Synthetic level-8 `stairs_down` cases verify:

| Final displacement | Integrated command path | Result |
|---:|---:|---|
| 1.2 m (a few steps, then stop) | 12 m | Demote to 7 |
| 1.2 m | 2 m | Stay at 8 |
| 4.1 m | 12 m | Promote to 9 |
| 0 m | 0 m | Stay at 8 |

Thus a few climbed steps do not accidentally promote; stopping despite a sustained travel command correctly demotes under the existing rule. No new curriculum flag or change was needed. The legacy Euclidean distance criterion can still promote a diagonal traversal before reaching a square tile edge, or promote after a traversal ending in a fall; this preset does not redefine that criterion.

## Validation results

`tools/test_stairs.py` passes synthetic reward/heading tests, GPU integration, independent force checks, first-fall/invalid trace-mask tests, CLI precedence/persistence, curriculum cases, and the default regression. Results: `logs/p25_checks/unit.json`; stdout: `logs/p25_checks/unit_final.log`.

| Check | Result |
|---|---|
| Terrain-relative clearance | At absolute sole height 0.30 m, gaps over 0/0.20 m treads are 0.30/0.10 m; target 0.10 m gives costs 0.20/0 at unit foot speed. Standing and stance-foot costs are zero. |
| Stumble | Above-threshold, exact-threshold, no-contact and negative-Fz synthetic cases give 1/0/0/1. Rotation, tangential friction, force sign and inactive contact masking pass. |
| Force validation | CPU MuJoCo pyramid decoding and an independent `Jᵀ force` free-root force projection agree with the new kernel. Standing feet support 322.317 N versus robot weight 327.077 N. |
| Sole geometry | Maximum absolute standing sole clearance 0.0004624 m. |
| Outward commands | All four pyramid types covered; positive forward/zero lateral commands, heading alignment, standing override and turn gating pass. Empirical share 0.49514 for requested 0.5. |
| Default configuration/observations | Serialized default environment config bytes match the original revision; initial observations and all observation groups at matched physical states/RNG states are byte-identical. Reset/resampled commands match exactly. |
| Independent physics rollouts | Not byte-identical, including repeated runs of the original code. In the final six-step check: original-vs-original maximum critic difference 1.14e-4; original-vs-new 9.50e-5; qpos difference 2.73e-6. These are recorded diagnostics, not a promise of deterministic GPU trajectories or a trajectory-tolerance test. |

**Installed-library finding:** MuJoCo-Warp's local `support.py` contact-force helper assumes consecutive pyramid rows (`_decode_pyramid`, around line 237). Its `constraint.py` builder assigns each row by an atomic increment (around lines 1746–1753), so row addresses can be nonconsecutive. The final test found 128 such contacts and a maximum 20.12 N discrepancy in that helper's per-contact world force. The new kernel gathers all four addresses explicitly. The independent CPU decoder and constraint-Jacobian checks validate that choice; the library itself was not modified.

The required scratch smoke run completed **1,024 environments × 24 steps × 300 iterations = 7,372,800 transitions**, seed 42, full terrain, proxies, preset enabled. All logged values and saved checkpoint tensors are finite; **zero blow-ups**. Learning-loop wall time: **206.87 s**, approximately 1.45 iterations/s. No curriculum promotions occurred.

| Iteration | Mean step reward | Episode return | Episode length (steps) | Mean level | XY tracking error (m/s) | Blow-ups |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.03149 | 2.93 | 89.94 | 0 | 0.6449 | 0 |
| 100 | 0.03775 | 5.58 | 138.11 | 0 | 0.5754 | 0 |
| 150 | 0.04549 | 17.09 | 371.94 | 0 | 0.4460 | 0 |
| 200 | 0.04795 | 29.52 | 614.49 | 0 | 0.3970 | 0 |
| 250 | 0.04803 | 32.87 | 682.11 | 0 | 0.3841 | 0 |
| 300 | 0.04752 | 32.54 | 686.31 | 0 | 0.3850 | 0 |

At iteration 300 the logged clearance/stumble weighted reward rates are -0.0006507/-0.0009611 (before multiplying by dt). Artifacts: `logs/p25_stairs_smoke/{train_log.csv,ckpt_000300.pt,last.pt,args.json}` and `logs/p25_checks/smoke_stdout.log`.

Resume checks, each with 1,024 environments:

- Original `logs/p2_full/ckpt_000200.pt` → iteration 203 **without** preset: finite, zero blow-ups, parameters updated. Saved environment config equals the normalized original config; runner config unchanged. Final mean reward 0.04916, XY tracking error 0.3765. Artifacts: `logs/p25_resume_default/`.
- Same original checkpoint → iteration 203 **with** preset: finite, zero blow-ups, parameters updated. Final mean reward 0.05009, tracking error 0.3632. Artifacts: `logs/p25_resume_preset/ckpt_000203.pt`.
- Preset checkpoint → iteration 204 with no preset argument: saved settings persist; finite, zero blow-ups, command-curriculum counter 4,896. Final mean reward 0.05149, tracking error 0.3606. Artifacts: `logs/p25_resume_preset/last.pt`.
- Copied the original four-row CSV into the preset test directory before training: all historical values remain identical, new reward columns are aligned, and four new rows append correctly. Original P2 artifacts were untouched.

Checkpoint/CSV assertions are saved in `logs/p25_checks/integration.json`. The short resume runs end before a normal episode completes, so their zero episode-return/length summaries are not learning outcomes.

The final synthetic diagnostic recorded **six falls**, two headings per level, with no blow-ups. Falls occur at 0.50–0.52 s, root x 1.058–1.096 m from the tile center, with knee/shin contact for 5–9 control steps. Peak foot horizontal forces are 123–164 N; peak shin normal loads 130–230 N. All terminate on **base height below 0.35 m**, with tilt 49–53°, rather than the tilt threshold. Four trials have one stumble-ratio event; two have none despite shin contact, illustrating why both signals are saved. These are deliberately forced default-pose failures, not teacher performance measurements. Artifacts: `logs/p25_checks/diagnostic/{trajectory.npz,summary.json,level*_trial*.png}`; all active/valid numeric samples checked finite and plots inspected. A separate 0.1 s P2 checkpoint run verifies the diagnostic's teacher-loading path only (`diagnostic_loader/`); its results carry no ascent-performance claim.

Local reproduction commands (from the repository root):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python tools/test_stairs.py
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python -u supermimic/train_ppo.py \
  --num_envs 1024 --iterations 300 --preset stairs_v1 --save_every 300 --log_dir logs/p25_stairs_smoke
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python -u supermimic/train_ppo.py \
  --num_envs 1024 --resume logs/p2_full/ckpt_000200.pt --iterations 203 \
  --save_every 203 --log_every 1 --log_dir logs/p25_resume_default
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python -u supermimic/train_ppo.py \
  --num_envs 1024 --resume logs/p2_full/ckpt_000200.pt --preset stairs_v1 --iterations 203 \
  --save_every 203 --log_every 1 --log_dir logs/p25_resume_preset
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python -u supermimic/train_ppo.py \
  --num_envs 1024 --resume logs/p25_resume_preset/ckpt_000203.pt --iterations 204 \
  --save_every 204 --log_every 1 --log_dir logs/p25_resume_preset
PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python supermimic/diagnose_stairs.py \
  --synthetic --trials 2 --seconds 8 --plots 6 --out logs/p25_checks/diagnostic
```

AST parsing and `git diff --check` pass. All generated tests, checkpoints, plots and caches remain inside the repository; generated run artifacts are ignored by Git.

## Cluster commands

Run from the repository root, with the requested interpreter available at the path below. The commands are supplied for execution on the cluster; no cluster connection was made. `full_s4` is an example existing teacher directory; use each final teacher's actual source directory for paired comparisons.

Fine-tune an existing walking teacher for **5,000 additional iterations** into a separate directory:

```bash
PY=/home/lxz/envs/sm_local/bin/python
TEACHER=logs/loco/full_s4/last.pt
TARGET=$(PYTHONDONTWRITEBYTECODE=1 "$PY" -c 'import sys, torch; print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["iter"]) + 5000)' "$TEACHER")
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 "$PY" -u supermimic/train_ppo.py \
  --resume "$TEACHER" --preset stairs_v1 --num_envs 4096 \
  --iterations "$TARGET" --save_every 1000 --log_every 50 \
  --log_dir logs/loco/full_s4_stairs_v1
```

Train from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python -u supermimic/train_ppo.py \
  --preset stairs_v1 --terrain full --collision proxy --seed 0 \
  --num_envs 4096 --max_init_level 0 --iterations 30000 \
  --save_every 1000 --log_every 50 --log_dir logs/loco/stairs_v1_scratch_s0
```

Matched evaluation before and after, including the other terrains to check retention:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python supermimic/eval_loco.py \
  --ckpt logs/loco/full_s4/last.pt --direction forward --trials 32
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python supermimic/eval_loco.py \
  --ckpt logs/loco/full_s4_stairs_v1/last.pt --direction forward --trials 32
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 /home/lxz/envs/sm_local/bin/python supermimic/diagnose_stairs.py \
  --ckpt logs/loco/full_s4_stairs_v1/last.pt --levels 7 8 9 --trials 16 \
  --plots 6 --out logs/p25_diagnostics/full_s4_after
```

`last_eval_fwd.json` contains separate rows for every terrain and level. Primary endpoint: mean forward `stairs_down` success over levels 7–9, with per-level results retained. Use the same checkpoint snapshot, trial count and seed for each ablation. A continued-training control without `--preset` is necessary because the original teachers already improve with more training.

For individual ablations, use the fine-tuning command above with distinct output directories and these replacements/additions:

| Experiment | Flags |
|---|---|
| Continued-training control | Omit `--preset` when resuming the original teacher |
| Outward commands only | `--outward_fraction 0.5` |
| Clearance only | `--foot_clearance_weight -0.25` |
| Stumble only | `--foot_stumble_weight -1` |
| Full preset without clearance | `--preset stairs_v1 --foot_clearance_weight 0` |
| Full preset without stumble | `--preset stairs_v1 --foot_stumble_weight 0` |
| Full preset without outward commands | `--preset stairs_v1 --outward_fraction 0` |

For an already preset-trained checkpoint, omitting `--preset` preserves its saved settings. Disable individual terms explicitly with zero to remove them. STOP/resume semantics are unchanged.

## Which changes helped?

Locally established: terrain-relative sole sampling distinguishes a foot over the raised tread from the same absolute foot height over the lower tread; stumble math detects the specified horizontal-force pattern; outward commands produce deliberate forward edge traversal commands on all four pyramid types; diagnostics capture and separate foot and shin contact; default compatibility and fine-tuning integration are tested.

**No reward or command change has yet been shown to improve final-teacher level-7–9 success.** The short scratch run validates numerical stability, not stair competence. Keep `stairs_v1` opt-in until paired cluster runs establish benefit over continued training, and use the ablations above to identify which terms help.
