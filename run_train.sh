#!/bin/bash
# Launch SuperMimic training on the local GPUs in a tmux session (data parallel via torchrun).
# Usage: [NGPU=8] [PY=python] ./run_train.sh <run_name> --motion_packs P.. [--loco_packs P..] [train.py args...]
RUN=${1:-run}
shift || true
NGPU=${NGPU:-8}
PY=${PY:-python}
cd "$(dirname "$0")"
mkdir -p "logs/$RUN"
tmux new-session -d -s "$RUN" \
  "$PY -m torch.distributed.run --standalone --nproc_per_node=$NGPU supermimic/train.py --log_dir logs/$RUN $* 2>&1 | tee -a logs/$RUN/stdout.log"
