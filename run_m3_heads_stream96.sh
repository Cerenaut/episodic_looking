#!/bin/bash
# Head baselines in the single-stream setting at the CLS/STM model's own budget.
#
# Why: the heads were only ever run to 12,000 exposures while the CLS/STM model was run to
# 96,000, so the published single-stream comparison sets a converged head against a model an
# eighth of the way through its budget. Checking the 12-epoch curves showed the "they have all
# plateaued anyway" assumption is only two-thirds true: FlyModel is flat from ~2,000 exposures
# and SDMLP from ~3,000, but NCM is still gaining 0.02-0.03 over its last three evaluations,
# and NCM is precisely the head the CLS/STM model overtakes at 96,000. The linear probe
# oscillates by up to 0.095 between adjacent points and needs seeds, not a longer run.
#
# So: all four heads, 3 seeds, 96 epochs (96,000 exposures), fine classes 3/4/5, batch 1.
# Written to a separate runs root so the existing 12,000-exposure results stay intact and both
# budgets remain quotable. Encodings are cached by the head script, so this is cheap.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CKPT=../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth
ROOT=runs_stream96
LOG=runs_local/stream96
mkdir -p "$LOG"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

step "heads at 96,000 exposures: 4 methods x 3 classes x 3 seeds"
for method in linear ncm flymodel sdmlp; do
  for fc in 3 4 5; do
    for seed in 0 1 2; do
      rp="$ROOT/cifar_100/head_${method}_streaming_[${fc}]_500/seed_$seed"
      if [ -s "$rp/results_streaming.txt" ]; then echo "skip (exists): $rp"; continue; fi
      step "$method class $fc seed $seed"
      $PY cifar_main_head_baselines.py --checkpoint "$CKPT" --coarse-classes 0 1 \
          --method "$method" --experiment-type streaming --fine-classes "$fc" \
          --epochs 96 --seed "$seed" --run-path "$rp" \
          >> "$LOG/${method}_c${fc}_s${seed}.log" 2>&1 || step "FAILED: $method c$fc s$seed"
    done
  done
done
step "ALL DONE"
