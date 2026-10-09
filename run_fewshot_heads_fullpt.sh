#!/bin/bash
# Few-shot heads with full pre-training (2026-09-27). Same as the heads part of run_m3_fewshot.sh, but with
# --pretrain-all-instances: the heads pre-train on all 500 images of fine-classes 1,2 per coarse class, as the
# STM checkpoints do, and only the fine-class-3 training set is cut to N. In runs_fewshot_heads/ the pre-training
# set was cut to N too. Same seeds, so each seed's fine-class-3 subset is identical to the old runs'.
# 12,000 exposures of fine class 3: epochs = 12000/(2N) = 375 / 93 / 12 for N = 16 / 64 / 500.
# Results runs_fewshot_heads_fullpt/; logs runs_local/fewshot_fullpt/.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
E13=../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth
LOG=runs_local/fewshot_fullpt
mkdir -p "$LOG"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
for N in 16 64 500; do
  EPOCHS=$(( 12000 / (2 * N) ))
  for m in linear ncm flymodel sdmlp; do
    [ -f "$LOG/head_${m}_$N.done" ] && continue
    step "head $m, N=$N ($EPOCHS epochs x $((2*N)) images), full pre-training"
    ( for s in 0 1 2; do
        $PY cifar_main_head_baselines.py --method $m --experiment-type continual --fine-classes 3 \
          --coarse-classes 0 1 --max-instances $N --pretrain-all-instances --epochs $EPOCHS --seed $s \
          --checkpoint $E13 --runs-root runs_fewshot_heads_fullpt || exit 1
      done && touch "$LOG/head_${m}_$N.done" && step "head $m N=$N done" || step "FAILED: head $m N=$N" ) > "$LOG/head_${m}_$N.log" 2>&1 &
  done
  wait
done
step "ALL DONE"
