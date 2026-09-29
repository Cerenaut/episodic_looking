#!/bin/bash
# Seeds 2 and 3 for the heads' few-shot selection at N = 1 and 4 (results_v3.tex, Hyperparameter selection; decided
# 29 Sep: at small N one seed is one draw of N images). Each unit copies its seed-1 counterpart in the same tree: the
# same --epochs (as last run, extensions included), learning rate and log-spaced evaluation, so metrics_v2.py averages
# the seeds epoch by epoch and extends them together. Resumable (run_v2.sh skips finished units).
# Usage: nohup bash seeds_v2.sh > runs_v2_pilot/seeds_nohup.log 2>&1 &
set -u
cd "$(dirname "$0")"
ROOT=runs_v2_pilot
LOG=$ROOT/seeds.log
step() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
step "seeds 2, 3 for the heads at N = 1, 4: code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty')"
for tree in $ROOT/linear_lr* $ROOT/ncm $ROOT/flymodel $ROOT/sdmlp_lr*; do
  name=$(basename "$tree"); model=${name%%_lr*}
  for n in 1 4; do
    for c in 3 4 5; do
      ref=$tree/fewshot/$model/pair0_1/seed1/fine${c}_n$n
      epochs=$(grep -h "start:" "$ref/job.log" 2>/dev/null | tail -1 | grep -o -- "--epochs [0-9]*" | awk '{print $2}')
      if [ -z "$epochs" ] || [ ! -f "$ref/job.done" ]; then step "SKIP $name N=$n fine $c: no finished seed-1 unit"; continue; fi
      lr=""; [ "$name" != "$model" ] && lr=${name#*_lr}
      for seed in 2 3; do
        if env RUNS="$tree" EPOCHS="$epochs" EVAL_POINTS=96 ${lr:+LR=$lr} bash run_v2.sh fewshot "$model" "0 1" "$seed" "$c" "$n" \
            >> "$tree.log" 2>&1 < /dev/null; then
          step "done   $name EPOCHS=$epochs seed $seed fine $c N=$n"
        else
          step "FAILED $name EPOCHS=$epochs seed $seed fine $c N=$n ($tree.log)"
        fi
      done
    done
  done
done
step "seeds finished"
