# Codex task P2.5 — make the teacher climb 17–20 cm stairs

Repo: /home/lxz/supermimic_vibed (branch `mjwarp-velocity`). Read `PERCEPTION_MILESTONES.md` (latest entries),
`tasks/codex_p2_report.md`, `supermimic/loco_env.py`, `supermimic/eval_loco.py`, `supermimic/terrain.py`.
Python: `/home/lxz/envs/sm_local/bin/python`; local RTX 4090 for tests. Only edit repo files; no commits, no
network, no new dependencies. Everything new must be behind `LocoCfg` options / CLI flags whose defaults keep the
current behavior exactly (teachers trained with the current code are being resumed on the cluster and must load
and behave identically).

## Problem

Forward evaluation (`eval_loco --direction forward`): teachers solve flat/rough/slopes/boxes and stair *descent*
at every level, but **ascending stairs fails above ~15 cm**: success 0.64 overall and 0.0 at levels 7–9
(step height 0.05 + 0.15 * level/9 m, tread 0.30 m). Stair ascent happens on the `stairs_down` tiles (the robot
spawns in the pit center and must climb out). Typical failure to check first: toe/shin catching the riser.

## What to add (each behind a flag; report which helped)

1. Diagnose: record a few level-7–9 ascent failures (trajectory of feet heights vs the stair profile, contact
   forces on feet/shins, where/why it falls). A quick look with the existing checkpoint
   `logs/p2_full/ckpt_000200.pt` is useless (it cannot walk) — instead build the diagnostic tool now and I will
   run it on the cluster checkpoints; for your own tests use synthetic cases.
2. Rewards commonly used for stairs (take terms/weights from the user's mjlab G1 configs in
   `/home/lxz/mjlab-loco/src/mjlab/tasks/velocity/` where they exist, cite them):
   - swing-foot clearance (foot height above the terrain under it during swing, target ~0.08–0.12 m, gated by
     command), using `HeightSampler` for the terrain under each foot;
   - foot stumble penalty (horizontal contact force on the feet large relative to the vertical one);
   - shin/knee contact penalty if not already covered by undesired contacts.
3. Command shaping on pyramid tiles (stairs and slopes, both signs): a configurable share of envs get a
   forward command pointing *outward* from the tile center (heading toward the nearest tile edge), so the
   curriculum sees deliberate forward climbing/descending instead of mostly sideways/backward commands.
4. Optional: separate curriculum level per terrain type is already logged; if promotion on `stairs_down` tiles is
   gated by the same distance rule, check that robots that climb a few steps and stop are not promoted/demoted
   wrongly.

Expose presets in `train_ppo.py` as `--preset stairs_v1` (all helpful changes on) plus individual flags, and make
`--resume` from an existing teacher checkpoint with a preset possible (fine-tuning a walking teacher with the new
terms), since that is the cheapest experiment on the cluster.

## Tests to run locally

- Unit checks of each new reward/command (synthetic states: foot above a riser vs on a tread, stumble force).
- A 1024-env, ~300-iteration smoke run with `--preset stairs_v1` from scratch: finite, no blow-ups.
- Resume a P2 checkpoint (`logs/p2_full/ckpt_000200.pt`) with `--preset stairs_v1` for a few iterations: loads,
  runs, and old-default behavior is unchanged when the preset is not given (byte-identical config/obs).

## Report back

`tasks/codex_p25_report.md`: changes, cited mjlab sources, test outputs, the diagnostic tool usage, and the exact
cluster commands for (a) fine-tuning existing teachers with the preset and (b) training from scratch. Do not
commit.
