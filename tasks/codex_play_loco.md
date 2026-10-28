# Interactive local play for loco policies on terrain

Write `tools/play_loco.py`: watch a trained loco checkpoint (teacher from `supermimic/train_ppo.py`, or elevation-map
student from `supermimic/train_student.py`, including the CNN map encoder) walk on the training terrain in the
MuJoCo interactive viewer, on this machine (RTX 4090, venv `/home/lxz/envs/sm_local`).

Requirements:
- Simulate with the training environment itself (`supermimic/loco_env.py` LocoEnv, MuJoCo-Warp on GPU, built from the
  checkpoint's `env_cfg` the same way `supermimic/eval_loco.py` does: curriculum/randomization off; for students the
  elevation map on, with `--map_noise on|off`, default on). Load the policy exactly as eval_loco does (`load_student`
  for students, the runner inference policy for teachers). Use few envs (1 is fine if LocoEnv allows it).
- Display: a CPU `mujoco.MjModel` of the same scene (same terrain hfield, filled via `write_hfield`) in
  `mujoco.viewer.launch_passive`; each control step copy env 0's qpos into it, `mj_forward`, `sync`, real-time pacing.
- Keyboard (viewer key_callback), reusing the key style of `tools/play_policy.py` where sensible: Up/Down forward
  speed +-0.1 m/s, PgUp/PgDn lateral, Left/Right yaw rate, Home zero command, Space pause, Backspace reset.
  Number keys or `[` `]` cycle the start tile type (`TILE_TYPES` in `supermimic/terrain.py`) and `-` `=` the difficulty
  level 0..9, then reset onto that tile (see how eval_loco sets `env.levels/columns/kinds` and resets).
  The env's own command resampling must not override the keyboard command (check how eval_loco keeps `env.cmd`).
- For students, optionally draw the elevation map the policy sees (valid cells) as small spheres/points via
  `viewer.user_scn`, toggle with `M`.
- CLI: `python tools/play_loco.py CKPT [--terrain stairs_down] [--level 8] [--map_noise on] [--device cuda:0]
  [--headless N]`; `--headless N` runs N steps without a window and prints distance travelled and whether it fell
  (so it can be tested without a display). Print the key help at start.
- Match the repo's terse style. Test with `--headless` on a local checkpoint (e.g. anything under `logs/` such as
  `logs/p4_checks`, or `logs/loco/student_cnn_stairs_scratchT/*.pt` if present). Do not touch the cluster, do not commit.
  Report: the exact command lines, headless results, anything that differs from training physics.
