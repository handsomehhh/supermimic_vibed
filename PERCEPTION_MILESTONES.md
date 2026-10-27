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

- 2026-10-27 — Started. Local test venv (`/home/lxz/envs/sm_local`, py3.13 + torch 2.9 + mujoco-warp 3.8) so Codex can test GPU code on the local RTX 4090.
