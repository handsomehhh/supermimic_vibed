Implemented opt-in `--observation depth`; map defaults and legacy checkpoints remain compatible. No cluster access or commits.

- **Camera:** 32×24, 87° FOV, G1 D435 position on `torso_link`, −50° pitch, 0.1–3 m axial depth, 30 Hz, 0–60 ms latency. Includes depth² Gaussian noise, pixel/edge dropout and extrinsic error. Terrain only; self-occlusion ignored.
- **4096-env timing:** 4.91 ms/frame; **4.04 ms/control step** including noise and refresh holding. Larger frames measured 11.12 ms at 48×36 and 18.76 ms at 64×48.
- **Training:** warmed iterations at 4096 envs took **1.38 s depth vs 1.62 s map (0.85×)**. Depth had zero blowups; the map comparison had six guarded blowups.
- **Tests:** CPU/GPU accuracy, orientation, latency/reset, CNN loading/export, and existing locomotion/map/stair regressions passed. Maximum reference error: **1.44 mm flat, 1.50 mm step**. ONNX error: **9.54e−7**.
- **Smoke:** six iterations at 256 envs, eval `--trials 1`, export, resume and PPO transfer completed. Eval had 70/70 falls after the brief training run; this validates integration, not walking quality.

Changed: `supermimic/{depth_camera,loco_env,student,train_student,train_ppo,eval_loco}.py`, `tools/{test_depth_camera,export_student,monitor_loco}.py`, and `README.md`.

[Full report, file descriptions and test evidence](/home/lxz/supermimic_vibed/logs/p5_checks/report.md) · [Deployment specification](/home/lxz/supermimic_vibed/logs/p5_checks/depth_export/student_spec.json)