# Codex task P1 — terrain in MuJoCo-Warp + height sampling

Repo: /home/lxz/supermimic_vibed (branch `mjwarp-velocity`). Read `README.md`, `supermimic/robot.py`,
`supermimic/env.py`, `supermimic/rotations.py` first and match their style (compact, few comments, type hints,
torch on GPU, `# fmt: skip` for long literal tables). Python env for running anything:
`/home/lxz/envs/sm_local/bin/python` (torch 2.9 CUDA, mujoco 3.8, mujoco_warp 3.8, warp 1.12; an RTX 4090 is
available). Robot assets are in `data/g1_description/` (`g1_29dof.xml` is built by `tools/urdf_to_mjcf.py`).
Do not add dependencies. Do not modify the motion-tracking code paths' behavior (all existing callers of
`sim_model()` and `TrackingSim` must behave exactly as before when no terrain is given).

## Deliverables

1. `supermimic/terrain.py`
   - `TerrainCfg` dataclass: tile size (default 8 m x 8 m), `num_rows` difficulty levels (default 10),
     `num_cols` tiles per level (default 20), horizontal resolution (default 0.05 m), vertical scale, border
     width (flat, e.g. 5 m), and the proportions of tile types:
     flat, rough (random uniform bumps, up to ~0.08 m at max difficulty), pyramid slope up and down (up to ~25°),
     pyramid stairs up and down (step height 0.05–0.20 m with difficulty, tread 0.30 m), and platforms /
     boxes (random rectangular blocks 0.05–0.25 m high). Difficulty increases linearly with the row index.
   - `generate(cfg, seed) -> Terrain` (numpy) returning: the height array `(H, W)` in meters, the world origin of
     each tile center `(num_rows, num_cols, 3)` with z = terrain height at the tile center, tile type ids, and
     metadata needed by the sampler (resolution, x/y offset).
   - `add_hfield(xml: str, terrain) -> xml` or equivalent so that `robot.sim_model(..., terrain=terrain)` replaces
     the floor plane by a MuJoCo `hfield` geom carrying the same heights (contype 0, conaffinity 1, like the
     current floor). Handle MuJoCo hfield conventions carefully (elevation data normalized to [0,1] times
     `size[2]`, base `size[3]`, row/column order and orientation) and verify them in the test below.
   - `HeightSampler` (torch, GPU): holds the height array as a tensor; `sample(xy: (..., 2)) -> (...)` bilinear
     world-frame height lookup (clamped to the map), and `scan(p, R, grid) -> (B, n)` that samples a
     robot-centric grid of points expressed in the yaw (heading) frame of the root, returning heights relative
     to the root height (`p_z - h`), e.g. the legged_gym grid 1.6 m x 1.0 m at 0.1 m (17 x 11 = 187 points,
     forward-biased is fine). Fully vectorized, no Python loops over envs.

2. `supermimic/robot.py`: `sim_model(timestep, terrain=None)`; unchanged output when `terrain is None`.

3. `tools/test_terrain.py` (run on the local GPU) that checks and prints:
   - sampler vs MuJoCo: for random xy, `HeightSampler.sample` equals the hfield surface height MuJoCo reports
     (e.g. `mj_ray` straight down on the CPU model, or a small sphere dropped and settled) within 1 cm;
   - contact sanity on every tile type with MuJoCo-Warp (not just CPU MuJoCo): spawn G1 envs (reuse
     `TrackingSim`-like setup or a minimal mujoco_warp loop with `robot.sim_model(..., terrain)`), standing in the
     default pose holding the default joint targets at the tile centers / on stair treads for ~1 s, and report
     the lowest robot point relative to the local terrain height (penetration must be < 2 cm) and that no world
     blows up (finite state). The project's `g1_29dof.xml` collision set is convex meshes + 8 small foot spheres;
     if mujoco_warp lacks a needed hfield collision pair, report exactly which pair and propose the smallest fix
     (e.g. extra sphere/capsule collision geoms on the feet/shins) — do not silently change collision geometry;
   - throughput: control steps/s for 4096 worlds on terrain vs on the flat plane (same settings as `env.py`:
     4 substeps of 5 ms, implicitfast, one CUDA graph per control step);
   - saves a top-down PNG of the generated heightfield to `/tmp/terrain_preview.png`.

## Report back

Write `tasks/codex_p1_report.md`: what you built, the hfield convention you verified, test outputs
(numbers), throughput, any mujoco_warp limitation found, and open issues. Keep code changes limited to the
files above (plus small helpers if needed). Do not commit.
