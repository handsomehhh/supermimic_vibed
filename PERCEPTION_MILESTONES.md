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
- 2026-10-28 — P2 (PPO teacher with privileged height scan, collision proxies, curriculum, eval) delegated to Codex.
- 2026-10-27 — Started. Local test venv (`/home/lxz/envs/sm_local`, py3.13 + torch 2.9 + mujoco-warp 3.8) so Codex can test GPU code on the local RTX 4090.
