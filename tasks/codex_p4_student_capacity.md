# P4: student capacity and stair-weighted student training

Context: the deployable elevation-map student (`supermimic/student.py`, `supermimic/train_student.py`) reaches robust
eval 0.98 overall but only ~0.65 on hard stair ascent (stairs_down tiles, levels 7-9), while its teacher
(`logs/loco/teacher_ft_outward_final.pt`, MLP [512,256,128]) gets 0.99. Map noise is ruled out (noise-off eval is
not better). Suspects: student capacity (MLP [256,128]; map encoder is a 2-layer MLP 128->64 over the flattened
map) and the terrain mix during distillation.

Implement, keeping every existing checkpoint loadable and the default behaviour unchanged:

1. `StudentModel` gets an optional convolutional map encoder, selected by a config key (e.g. `map_encoder: "mlp" | "cnn"`,
   default "mlp"). For "cnn": reshape the elevation-map part of the observation into image channels
   (heights and the validity mask as separate channels; check `supermimic/elevation_map.py` / `loco_env.py`
   for the exact flattened layout and its row/column order, do not guess), e.g. 3 conv layers (16-32-32 channels,
   3x3, ELU, one stride-2), flatten, Linear -> `map_latent`. Keep `StudentNormalization` semantics (metric heights and
   the mask are not normalized).
2. `ExportStudent` and `tools/export_student.py` must work for both encoders (TorchScript and ONNX), and
   `load_student` in `supermimic/student.py` (used by `supermimic/eval_loco.py`) must rebuild the right architecture
   from the checkpoint's stored config.
3. `train_student.py` new flags (new distillation runs only; PPO stage inherits from the student checkpoint):
   `--map_encoder {mlp,cnn}`, `--hidden_dims` (e.g. `512 256 128`), `--map_latent`.
4. `train_student.py --terrain_proportions` (7 weights in `TILE_TYPES` order, `supermimic/terrain.py`), applied to the
   env terrain for both stages, stored in the checkpoint's env_cfg as usual. Check how `train_ppo.py --preset stairs_v1`
   changes the terrain and reuse that mechanism if it fits.
5. Tests: extend or add `tools/test_*.py` covering both encoders (forward shape, export round-trip equals eager output,
   load_student from a saved checkpoint), and run the existing student/eval tests. Then run a short local smoke test on
   this machine (RTX 4090, venv `/home/lxz/envs/sm_local`): a cnn distillation for a few iterations with ~256 envs,
   a PPO stage from that checkpoint for a few iterations, `eval_loco.py --ckpt ... --trials 1`, and export.

Constraints: match the surrounding code style (terse, few comments). Do not change the defaults of existing flags.
Do not touch the cluster and do not commit. Write a short report: changed files, the observation layout you found,
parameter counts (mlp default vs cnn with `--hidden_dims 512 256 128`), test and smoke results.
