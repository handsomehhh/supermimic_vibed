# Codex task P2 — PPO velocity teacher on terrain (privileged height scan)

Repo: /home/lxz/supermimic_vibed (branch `mjwarp-velocity`). P1 is merged: `supermimic/terrain.py`
(`TerrainCfg`, `generate`, `HeightSampler`), `robot.sim_model(timestep, terrain=None)`. Read `README.md`,
`PERCEPTION_MILESTONES.md`, `tasks/codex_p1_report.md`, `supermimic/env.py`, `supermimic/robot.py`,
`supermimic/terrain.py` and match the code style (compact, typed, vectorized torch on GPU, one CUDA graph per
control step, `use_warp_stream`). Python: `/home/lxz/envs/sm_local/bin/python` (torch 2.9 CUDA, mujoco 3.8.0,
mujoco_warp 3.8.0, warp 1.12, rsl-rl-lib 5.2.0 with TensorDict obs groups and `OnPolicyRunner` /
`DistillationRunner`). A local RTX 4090 is available — run your GPU tests yourself. Cluster training runs on
H100s with the same package versions, one run per GPU. Do not add dependencies; do not change the
motion-tracking code paths (`env.py`, `train.py`, `evaluate.py` must behave exactly as before). Only edit files
inside the repo; no git commits, no network use.

Reference for proven reward terms, weights, curriculum and PPO hyperparameters for the G1 on rough terrain:
the user's own mjlab fork at `/home/lxz/mjlab-loco/src/mjlab/tasks/velocity/` (`velocity_env_cfg.py`,
`mdp/rewards.py`, `config/g1/`, `rl/`). Read it, reuse its choices where sensible, and cite what you took.

## 1. Collision proxies (robot.py)

`sim_model(..., collision="mesh" | "proxy")`, default `"mesh"` (unchanged). `"proxy"` replaces the robot's
convex-mesh collision geoms by primitives sized from the meshes (capsules for thigh/shin/upper and lower arm,
boxes or capsules for pelvis/torso, a box per foot sole or keep the foot spheres but sized realistically) so
that hfield contacts are cheap and avoid the hfield–mesh 50-contacts-per-pair cap found in P1. Keep masses and
inertias unchanged (only collision geoms change). Report throughput (4096 worlds, terrain) for mesh vs proxy and
re-run the P1 standing/penetration check with proxies on every tile type on the GPU.

## 2. `supermimic/loco_env.py` — `LocoEnv` (rsl_rl `VecEnv`)

- G1 on the P1 terrain, N envs on one GPU, 50 Hz control, 4 x 5 ms substeps, same PD actuators and per-joint
  action scale as `env.py` (`action_scale`, `default_joint_pos_list`); setpoint clamp to the joint range and the
  blow-up guard from `env.py` (roll back + reset + exclude).
- Terrain curriculum (Isaac Lab style): each env owns a (level, column) tile, spawns at the tile center with
  random yaw; on reset move up a level if it walked more than half a tile, down if it walked less than half of
  the commanded distance; envs that solve the top level are re-randomized over all levels. Initial levels
  0..`max_init_level`.
- Commands resampled every ~10 s: vx, vy, yaw rate ranges like the mjlab G1 rough config (plus a share of
  standing commands and a heading-command mode if mjlab uses one).
- Observation groups (TensorDict): `policy` = base angular velocity, projected gravity, command, q - q0, qd,
  last action (with the usual noise); `height_scan` = privileged `HeightSampler.scan` (187 points, clipped) —
  the teacher's actor uses `policy + height_scan`; `critic` = everything above noise-free plus base linear
  velocity and friction. Keep the scan grid configurable (P3 will add a noisy elevation-map group).
- Rewards: velocity tracking (lin xy, yaw), and the regularizers/penalties of the mjlab G1 rough config
  (z velocity, roll/pitch rate, torques/acc/action rate, feet air time for a biped, feet slide, undesired body
  contacts, joint limits, joint deviation of hips/arms/waist, orientation, base height relative to the terrain
  below the base, termination penalty). Contacts: derive per-body contact (and force if needed) from
  mujoco_warp contact data; document how.
- Terminations: fall (base too low relative to terrain or too tilted, or non-foot body contact if mjlab uses
  it), time out (20 s, bootstrapped).
- Domain randomization: friction, base mass / CoM, push perturbations, observation noise.
- `extras["log"]`: mean terrain level, mean level per terrain type, tracking errors, reward terms.

## 3. `supermimic/train_ppo.py`

rsl_rl `OnPolicyRunner` with an mjlab-like config (actor/critic MLPs, adaptive lr, entropy, steps per env,
epochs, minibatches, gamma, lambda). Args: `--num_envs`, `--iterations`, `--seed`, `--log_dir`, `--collision`,
`--resume`. Log a `train_log.csv` row every 50 iterations (iteration, wall time, mean reward, episode length,
mean terrain level, per-type levels, tracking errors, it/s) and checkpoints every N iterations, like
`supermimic/train.py` does, so the existing orchestrator conventions (STOP file → save and exit; `last.pt` for
resume) still apply — implement the STOP-file behavior.

## 4. `supermimic/eval_loco.py`

Deterministic evaluation of a checkpoint: for each terrain type and difficulty level, spawn envs at tile
centers, command forward walking (and backwards for "down" variants) at 0.6 m/s for 8 s, report success
(reached > 3 m from the start without falling), mean tracking error, falls. Write `eval_loco.csv` /
`*_eval.json` next to checkpoints and support `--watch` like `supermimic/evaluate.py`.

## 5. Tests you must run locally (GPU) and report

- 2 short training runs (1024 envs, ~200 iterations each): flat-only terrain and full terrain; show the reward
  and terrain level going up, no NaNs, no blow-ups.
- throughput (env steps/s incl. rewards/obs) for 4096 envs, mesh vs proxy.
- eval_loco on the 200-iteration checkpoint (numbers will be low; it must run).

## Report back

Write `tasks/codex_p2_report.md`: design decisions and what you took from mjlab (with file/line references), the
contact extraction method, proxy geometry table, test outputs, throughput, and the exact command to launch a
full training run. Do not commit.
