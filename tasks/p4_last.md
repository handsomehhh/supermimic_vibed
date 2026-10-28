Implemented with defaults preserved. Existing MLP distillation/PPO checkpoints and default initialization remain bit-identical.

Changed files:

- [student.py](/home/lxz/supermimic_vibed/supermimic/student.py): CNN encoder and checkpoint architecture reconstruction.
- [train_student.py](/home/lxz/supermimic_vibed/supermimic/train_student.py): architecture flags and terrain weights for both stages/resume.
- [export_student.py](/home/lxz/supermimic_vibed/tools/export_student.py): both encoder export paths.
- [test_student_model.py](/home/lxz/supermimic_vibed/tools/test_student_model.py): layout, normalization, checkpoint, and export tests.

Observation layout is `[history, all heights, all validity values]`, with **x outer, y inner**. The default CNN input is `[batch, 2, 25, 15]`; heights and masks remain unnormalized.

Deployable parameter counts, with history length 5 and map latent 64:

| Encoder / policy widths | Parameters |
|---|---:|
| Default MLP / `[256,128]` | 280,541 |
| CNN / `[512,256,128]` | 674,253 |

All new tests and existing student map/monitor/lifecycle tests passed. Local RTX 4090 smoke completed: 3 distillation iterations, 3 PPO iterations, and one resumed distillation iteration at 256 environments, with zero blowups. Architecture inheritance and terrain persistence were verified.

Both default and robust-forward evaluations completed with `--trials 1`: **0/70 successes** after the short run. TorchScript matched eager exactly; maximum ONNX error was `7.16e-7`.

The named teacher was absent locally; smoke used `logs/p25_stairs_smoke/last.pt`. [Smoke evidence](/home/lxz/supermimic_vibed/logs/p4_checks/smoke_verification.json) and exports are under `logs/p4_checks/cnn_ppo/export/`.

No cluster access or commits.