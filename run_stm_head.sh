#!/bin/bash
# Supervised STM head (cifar_main_head_baselines.py --method stm): the RL STM's network as a supervised classifier on the
# frozen e13 LTM's 512-d features; pair A, seed 1, pre-training on fine 1,2 (12 epochs) then continual 3,4,5 (12 epochs
# each), validation split 100/0. lr 0.1 momentum 0.5 = the RL STM's optimizer (primary); 0.03 and 0.01 as sensitivity.
# Usage (repo root): bash run_stm_head.sh
set -uo pipefail
for lr in 0.1 0.03 0.01; do
  D=runs_cond/heads/stm/lr$lr/pair0_1/seed1/order3_4_5
  [ -f $D/job.done ] && continue
  mkdir -p $D; { git rev-parse HEAD; git status --porcelain --untracked-files=no | sed 's/^/dirty: /'; } > $D/code_commit.txt
  echo "[$(date '+%F %T')] start stm head lr $lr"
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ../.venv/bin/python cifar_main_head_baselines.py --method stm --experiment-type continual \
    --fine-classes 3 4 5 --coarse-classes 0 1 --seed 1 --checkpoint ../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth \
    --val-holdout 100 --split-seed 0 --pretrain-epochs 12 --lr $lr --pretrain-lr $lr --momentum 0.5 --pretrain-momentum 0.5 \
    --device cpu --run-path $D > $D/job.log 2>&1 && touch $D/job.done && echo "[$(date '+%F %T')] done lr $lr" || echo "FAILED lr $lr"
done
