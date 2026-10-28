# P5: depth-camera student (alternative to the elevation-map student)

Context: the deployable elevation-map student works (v2: `student_cnn_stairs_scratchT`, CNN map encoder,
`supermimic/student.py`, `supermimic/train_student.py`, `supermimic/elevation_map.py`, `supermimic/loco_env.py`).
Add a second deployable observation: a single forward-down head depth camera, distilled from the same teachers.

Implement, keeping all existing behaviour, checkpoints and defaults unchanged:

1. `supermimic/depth_camera.py`: synthetic depth images rendered on the GPU in torch by ray-marching the terrain
   height field (`supermimic/terrain.py` `HeightSampler.sample`), batched over all envs. Only the terrain is rendered
   (robot self-occlusion ignored; say so in the docstring). Config dataclass like `ElevationCfg`: resolution
   (default 64x48 is too slow? measure; aim for something like 32x24 or 48x36 at 4096 envs), horizontal FOV ~87 deg,
   mount on the torso/head link with pitch ~ -50 deg (look up a link name in the G1 model used by `robot.py`), near/far
   clip 0.1/3.0 m, refresh ~ 30 Hz with latency, and a noise model typical of stereo depth: per-pixel Gaussian noise
   growing with range squared, random pixel dropout/holes, edge dropout, small extrinsic error. Use ray-marching
   with a coarse step then bisection refinement; report accuracy against a fine-step reference in a test.
2. Wire it into `LocoEnv` as an optional observation group (e.g. `depth_image`), mutually independent of
   `elevation_map`, enabled via config (new distillation runs only), with history/latency handled like the
   elevation map.
3. Student: a CNN depth encoder variant in `StudentModel` (selected via config, e.g. inputs `policy_history`,
   `depth_image`), export (TorchScript + ONNX) and `load_student` support, eval support in `supermimic/eval_loco.py`
   (it must build the env with the checkpoint's observation config). Add `train_student.py --observation {map,depth}`
   (default map).
4. Tests in `tools/test_*.py` (ray-march accuracy vs fine reference on flat ground and a step, image orientation:
   near ground at the bottom of the image, export round trip), existing tests still pass. Local smoke on this machine
   (RTX 4090, venv `/home/lxz/envs/sm_local`): depth distillation a few iterations at 256 envs, eval `--trials 1`,
   export. Report rendering time per step at 4096 envs (estimate from 256/1024 if needed) and pick the default
   resolution so a training iteration is at most ~1.5x the map student's.

Constraints: match the surrounding style (terse, few comments). Do not change existing defaults. Do not touch the
cluster and do not commit. Report: changed files, camera config, render time, test and smoke results.
