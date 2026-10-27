#!/bin/bash
# P2 batch 1: one PPO teacher per GPU plus its eval watcher on the same GPU.
# Usage (cluster, repo root): tools/launch_loco_batch.sh
cd "$(dirname "$0")/.."
PY=${PY:-/data_zcy/lxz/env/sm_venv/bin/python}
run() {  # gpu name args...
  g=$1; n=$2; shift 2
  d=logs/loco/$n; mkdir -p "$d"
  CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 nohup "$PY" supermimic/train_ppo.py --num_envs 4096 --iterations 30000 \
    --save_every 1000 --log_dir "$d" "$@" >> "$d/stdout.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=$g PYTHONUNBUFFERED=1 nohup "$PY" supermimic/eval_loco.py --run_dir "$d" --watch >> "$d/eval_stdout.log" 2>&1 &
}
run 0 full_s0 --terrain full --collision proxy --seed 0
run 1 full_s1 --terrain full --collision proxy --seed 1
run 2 full_s2 --terrain full --collision proxy --seed 2
run 3 full_s3 --terrain full --collision proxy --seed 3
run 4 flat_s0 --terrain flat --collision proxy --seed 0
run 5 full_mesh_s0 --terrain full --collision mesh --seed 0
run 6 full_init4_s0 --terrain full --collision proxy --seed 0 --max_init_level 4
run 7 full_init4_s1 --terrain full --collision proxy --seed 1 --max_init_level 4
