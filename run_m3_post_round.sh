#!/bin/bash
# Post-round evaluations on the M3, now that the single-stream run has freed the GPU.
# 1. The single-stream final checkpoints under BOTH bias modes. This is the measurement
#    the run was restarted to make possible: it settles whether to report single-stream
#    sampled (comparable with figure 10) or at the policy mean (our standard elsewhere)
#    on the numbers rather than on the rough 0.06 offset assumed in the notes.
# 2. Pre-continual baselines for the ten seeded checkpoints, which BWT_0 and FWT need.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
E13=$CK/cifar_100_subclasses_12_e13.pth
LOG=runs_local/post_round
mkdir -p "$LOG"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

step "=== 1. single-stream final checkpoints, sampled vs mean ==="
for c in 3 4 5; do
  for mode in sampled mean; do
    tag="stream_c${c}_${mode}"
    [ -f "$LOG/$tag.done" ] && continue
    if [ "$mode" = mean ]; then bias="--eval-bias mean"; steps=800; else bias=""; steps=1600; fi
    step "evaluating stream_full_c$c at $mode bias"
    $PY cifar_main_stm_training.py --experiment-type evaluate --fine-classes 3 4 5 --coarse-classes 0 1 \
        --evaluate-steps $steps $bias --ltm-checkpoint $E13 \
        --stm-checkpoint $CK/variants/stm_stream_full_c$c.pth --run-root "$LOG/runs_$tag" \
        > "$LOG/$tag.log" 2>&1 && touch "$LOG/$tag.done" || step "FAILED: $tag"
  done
done

step "=== 2. pre-continual baselines for the ten seeded checkpoints ==="
for v in pt12 pt40; do
  for s in 1 2 3 4 5; do
    tag="baseline_${v}_s${s}"
    [ -f "$LOG/$tag.done" ] && continue
    ck=$CK/variants/stm_rl_fixed_e13_${v}_seed${s}.pth
    [ -f "$ck" ] || { step "MISSING checkpoint: $ck"; continue; }
    step "baseline $v seed $s"
    $PY cifar_main_stm_training.py --experiment-type evaluate --fine-classes 3 4 5 --coarse-classes 0 1 \
        --evaluate-steps 800 --eval-bias mean --ltm-checkpoint $E13 \
        --stm-checkpoint "$ck" --run-root "$LOG/runs_$tag" \
        > "$LOG/$tag.log" 2>&1 && touch "$LOG/$tag.done" || step "FAILED: $tag"
  done
done
step "ALL DONE"
