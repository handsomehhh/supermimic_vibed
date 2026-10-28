Added [tools/play_loco.py](/home/lxz/supermimic_vibed/tools/play_loco.py) with teacher/MLP/CNN loading, all requested controls, tile resets, and the `M` elevation-map overlay.

Exact headless commands run:

```bash
/home/lxz/envs/sm_local/bin/python tools/play_loco.py logs/loco/student_cnn_stairs_scratchT/ckpt_005064.pt --terrain stairs_down --level 8 --map_noise on --device cuda:0 --headless 400
/home/lxz/envs/sm_local/bin/python tools/play_loco.py logs/loco/student_cnn_stairs_scratchT/ckpt_005064.pt --terrain stairs_down --level 8 --map_noise off --device cuda:0 --headless 400
/home/lxz/envs/sm_local/bin/python tools/play_loco.py logs/p2_flat/last.pt --terrain flat --level 0 --device cuda:0 --headless 650
/home/lxz/envs/sm_local/bin/python tools/play_loco.py logs/p3_student/last.pt --terrain flat --level 0 --map_noise on --device cuda:0 --headless 400
```

| Checkpoint | Steps | Distance¹ | Fell | Blowups |
|---|---:|---:|---|---:|
| CNN, noise on | 400 | 4.443 m | False | 0 |
| CNN, noise off | 400 | 4.207 m | False | 0 |
| Teacher | 650 | 0.019 m | False | 0 |
| MLP student | 400 | 0.013 m | False | 0 |

¹ Maximum XY displacement before the first fall. The early teacher/MLP checkpoints barely moved.

Remove `--headless N` to launch the viewer. Live viewer, keyboard state, command persistence, heightfield identity, and map-toggle checks passed using:

```bash
timeout 60s /home/lxz/envs/sm_local/bin/python -u /tmp/play_loco_check.py
timeout 60s /home/lxz/envs/sm_local/bin/python -u /tmp/play_loco_timing.py
```

Physics stays in the training GPU environment with unchanged substeps, collisions, and actuators. Evaluation differences match `eval_loco`: randomization/curriculum/proprioceptive noise disabled, friction 1.0 instead of training terrain friction 0.8, and the canonical seven-type terrain layout.

**Performance limitation:** pacing targets 50 Hz, but the full-terrain viewer achieved roughly 7–9 updates/s during testing. Headless environment steps averaged about 8.4 ms.

No cluster access or commits.