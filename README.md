# SuperMimic reproduction on MuJoCo-Warp

**Unofficial** reproduction of *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World
Model* (arXiv 2610.28480) — no official code is released, so everything here is implemented from
the paper. Target experiment: the controlled LAFAN1 comparison (Fig. 4a): 40 G1-retargeted LAFAN1
clips (2.45 h), one GPU; the paper reaches evaluation normalized return 0.82 after 1.2 h on an H100.

Author: **Chong Zhang** (ETH Zurich). License: [MIT](LICENSE).

> **Disclaimer:** this repository was speed-vibecoded in 1 hour with an AI coding
> assistant. It
> deliberately does not live under any lab's organization, because the code is too much "slop"
> for that: fast and unreviewed. Use it at
> your own risk.

> **One G1 motion tracker for all 40 LAFAN1 clips, trained in only ~1.5 GPU-hours**
> (1.47 h on a single RTX PRO 6000): evaluation return **0.829** with **0/40 failed clips**.
> For scale, the paper's PPO-in-simulator baseline does not reach return 0.82 within 18 h on up to
> 8 GPUs.

![SuperMimic G1 tracking LAFAN1 motions](results/supermimic_lafan1_highlight.gif)

*Solid robot: the trained policy (Isaac Lab version) simulated with MuJoCo-Warp. Blue ghost: the reference
motion. Clips: dance, jumps, fall-and-get-up, sprint.*

Progress and timestamps: [MILESTONES.md](MILESTONES.md).

## Status of this branch

The original reproduction (commit `a1508d4`, results above) ran on Isaac Lab 3 kit-less with the
Newton/MuJoCo-Warp backend. This branch drops Isaac Lab and drives **MuJoCo-Warp directly**, and adds
the paper's **velocity-command interface** (App. B, G): a velocity encoder trained by RL through the
same world model, jointly with motion tracking. Training is data parallel over the local GPUs.

## Layout

| Path | What |
|---|---|
| `data/` | not tracked: G1 URDF/meshes and generated `g1_29dof.xml`; LAFAN1 CSVs; motion packs (`data/packs/*.pt`) |
| `supermimic/` | implementation (see below) |
| `tools/` | asset conversion, motion packing, simulator test, experiment orchestrator, plotting, rendering |
| `logs/<run>/` | not tracked: `train_log.csv`, `eval.csv`, checkpoints, stdout |
| `results/` | success curve and highlight GIF of the Isaac Lab run |

## Setup

Requirements: Linux, NVIDIA GPU(s), Python 3.12+ with `torch` (2.9 tested), `mujoco` and
`mujoco-warp` (3.8 tested), `warp-lang`, `numpy`, `matplotlib`; `gcc` for `torch.compile`.

```bash
tools/download_data.sh    # LAFAN1 G1 retargets + G1 URDF/meshes, MJCF, LAFAN1 packs
```

Motion packs (`tools/pack_motions.py`) hold root position, root quaternion (wxyz) and the 29 joint
angles in Unitree SDK order at 50 Hz. Sources are LAFAN1 CSVs (30 fps, resampled) or BeyondMimic-style
npz files (`fps`, `joint_pos`, `body_pos_w`, `body_quat_w`, ... in Isaac Lab breadth-first joint order;
the order is verified by FK against `body_pos_w`). `--filter loco` keeps walking/jogging/running clips
by name (props, injuries, dances, stylizations excluded) whose pelvis stays upright:

```bash
python tools/pack_motions.py --npz_root <motion_data_used_g1>/sonic_filtered --filter loco --out data/packs/sonic_loco.pt
python tools/pack_motions.py --npz_root <motion_data_used_g1> --out data/packs/all_g1.pt
python tools/test_env.py --packs data/packs/lafan_loco.pt     # FK vs simulator, playback, throughput
```

## Usage

```bash
# motion tracking + velocity interface on 8 GPUs
NGPU=8 ./run_train.sh loco --motion_packs data/packs/lafan_loco.pt --loco_packs data/packs/lafan_loco.pt
python supermimic/evaluate.py --run_dir logs/loco --watch          # scores every checkpoint into eval.csv
# the three experiments in sequence, with automatic stopping (status in logs/orchestrator.json)
nohup python tools/orchestrate.py > logs/orchestrator.log 2>&1 &
```

Checkpoints are written every 2000 iterations. Creating `logs/<run>/STOP` makes the trainer save a
checkpoint and exit; `--resume logs/<run>/last.pt` continues a run.

## Implementation (paper section -> file)

* G1 29-DoF, BeyondMimic actuators (10 Hz natural frequency, damping ratio 2, motor armatures,
  action scale 0.25·τmax/Kp) as MuJoCo position actuators — `supermimic/robot.py`.
* Simulator: MuJoCo-Warp, 50 Hz control, 4 substeps of 5 ms (one CUDA graph per control step),
  implicitfast, pyramidal cones; per-world friction and torso CoM — `supermimic/env.py`.
* Reference library: packs at 50 Hz, finite-difference velocities, FK of 14 tracked bodies (cached for
  libraries up to 20 M frames, computed on demand above), 1 s bins, per-clip motion/loco tags and the
  reference velocity command (25-frame centered mean) — `supermimic/motion.py`, FK `supermimic/kinematics.py`.
* MDP (App. A): observation (q−q0, q̇, IMU ω, projected gravity, a_{t−1}; 10-frame history; Table A.1
  noise), BeyondMimic tracking reward without the self-collision term and joint-limit weight −2,
  termination |e_z|>0.25 m or e_tilt>0.8, sim context ξ = (friction U[0.3,1.2], joint offsets
  U[±0.01], torso CoM offset). Velocity interface (Table A.6): heading-frame velocity and yaw-rate
  tracking, action smoothness, joint limits, torso sideways lean, torso roll/pitch rate; fall
  termination h<0.35 m or 1+u_z>0.7 — `supermimic/mdp.py`.
* Return-based reset sampling for motion envs (50% uniform / 40% softmax(−25 G_b/G_max) / 10% past
  softmax); velocity envs reset uniformly over the locomotion library and resample their command
  source at reset and every 5 s (50% reference command, else uniform in vx∈[−1.5,2], vy∈[−1.32,1.32],
  ωz∈[−3.35,3.35], zero with probability 0.1) — `supermimic/env.py`.
* World model f_ω: MLP(2048,2048,2048) on heading-invariant state features + action + ξ, predicting
  frame-local increments; fit on k=8-step autoregressive windows (Eq. 1) — `supermimic/networks.py`.
* Policy (Eq. A.1): motion encoder MLP(2048,1024,512,256) on K=10 future reference frames, velocity
  encoder MLP(256,256) on the normalized command and the observation history, z∈R^32 (tanh), shared
  decoder MLP(2048,1024,512,256). Half of the environments and of each imagined batch use the velocity
  interface; its return gradient reaches the velocity encoder through the decoder with a stop-gradient
  on the decoder weights, plus a 0.1-weighted MSE pulling z_vel toward sg(z_motion) on rollouts with
  reference commands. One critic MLP(256,128) per interface on GAE(λ=0.95) targets.
* Hyperparameters as Table A.5 (per GPU): 4096 envs, buffer 1000 steps/env, action noise N(0, 0.1²),
  UTD 0.4, 10 WM batches of 2048 windows, 4096 imagined rollouts, Adam lr 2e-4·√(#GPUs) (policy
  β=(0.9,0.95)), cosine decay to 0.03× over 250k iterations, grad clip 1.0 (+ element-wise 0.1 for the
  policy), γ=0.99. Multi-GPU: gradients and normalizer statistics averaged across ranks.
* Evaluation — `supermimic/evaluate.py`: motion tracking (App. F.1) on up to 1024 motion clips and
  512 locomotion clips; velocity step test (App. G.2) on 672 commands × 4 trials from standing.

## Deviations / choices not specified by the paper

* **Self-collisions disabled** in the simulator (speed); robot–ground contacts use the URDF's
  convex mesh hulls and foot spheres.
* **bf16 autocast for MLP matmuls** (paper: TF32); state integration, FK and rewards stay fp32.
* Reward, observation and termination heads are **analytic** (allowed by §3.2); termination in
  imagination is a sigmoid-softened version of the hard thresholds.
* The motion-encoder goal additionally contains the reference root orientation relative to the robot
  root (6D, BeyondMimic's `motion_anchor_ori_b`); the proprioceptive observation shared with the
  velocity interface stays as in Table A.1.
* The velocity encoder's "feature of the last 10 frames of o" is the normalized observation history.
* 500 world-model-only warm-up iterations before policy updates start.
* Torso-CoM and friction are randomized once per environment at start-up; joint offsets are resampled
  every episode. Episodes time out after 10 s or at the clip end (bootstrapped, not terminal).

## Attribution and licenses

* Method: Q. Liao, Y. Shao, J. Long, K. Zakka, R. Horowitz, P. Abbeel, K. Sreenath,
  *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*,
  CoRL 2026, arXiv:2610.28480. This code is an independent re-implementation, not affiliated with
  the authors.
* Isaac Lab (BSD-3-Clause), Newton and MuJoCo-Warp (Apache-2.0) are used unmodified.
* LAFAN1 motions (Ubisoft La Forge): CC BY-NC-ND 4.0; G1 retargets and description by Unitree
  (BSD-3-Clause). The motion data is not redistributed here; the demo GIF shows the
  reference motions for illustration only and is not covered by this repository's MIT license.

## License

The code in this repository is released under the [MIT License](LICENSE).

Copyright (c) 2026 ETH Zurich, Chong Zhang.
