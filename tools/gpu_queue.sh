#!/usr/bin/env bash
# Launch queued loco runs onto GPUs that stay idle (util < 5% for 3 polls). Queue file: one run per line,
# "<run_name> <train script and args>"; launched lines move to <queue>.done. Each run gets its eval watchers.
QUEUE=${1:-logs/loco/queue.txt}
PY=/data_zcy/lxz/env/sm_venv/bin/python
declare -A idle
while true; do
  i=0
  for u in $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits); do
    if [ "$u" -lt 5 ]; then idle[$i]=$(( ${idle[$i]:-0} + 1 )); else idle[$i]=0; fi
    if [ "${idle[$i]}" -ge 3 ] && [ -s "$QUEUE" ]; then
      line=$(head -1 "$QUEUE"); sed -i 1d "$QUEUE"; echo "$line" >> "$QUEUE.done"
      d=${line%% *}; args=${line#* }
      echo "$(date -u +%FT%TZ) gpu $i <- $d"
      CUDA_VISIBLE_DEVICES=$i nohup $PY $args --log_dir logs/loco/$d > logs/loco/$d.out 2>&1 < /dev/null &
      sleep 2
      CUDA_VISIBLE_DEVICES=$i nohup $PY supermimic/eval_loco.py --run_dir logs/loco/$d --watch --map_noise on > logs/loco/$d.eval.out 2>&1 < /dev/null &
      CUDA_VISIBLE_DEVICES=$i nohup $PY supermimic/eval_loco.py --run_dir logs/loco/$d --watch --map_noise on --trials 32 --direction forward --yaw_deg 30 --robust > logs/loco/$d.evalrb.out 2>&1 < /dev/null &
      idle[$i]=-5  # give the new run time to load before it can count as idle again
    fi
    i=$((i + 1))
  done
  sleep 60
done
