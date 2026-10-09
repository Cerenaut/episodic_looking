#!/bin/bash
# Continual confirmation runs of the v2 hyperparameter selection (results_v2.tex, Hyperparameter selection): each
# continual choice re-run at its selected budget (EPOCHS per phase), 3 orders, pair A, seed 1, validation split, into
# runs_v2_pilot_confirm/<tree>/. Heads in one lane, LTM-only in another. Resumable (run_v2.sh skips finished units).
# The criterion (agreed 28 Sep): the run's selection-curve value at the budget within 0.03 of the selection run's
# smoothed value there; checked afterwards with metrics_v2.py.
# Usage: nohup bash confirm_v2.sh > runs_v2_pilot_confirm/nohup.log 2>&1 &
set -u
cd "$(dirname "$0")"
ROOT=runs_v2_pilot_confirm
mkdir -p "$ROOT"
LOG=$ROOT/confirm.log
step() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
# tree | model | EPOCHS | LR (empty: none)
CHOICES_HEADS="linear_lr0.001|linear|8|0.001 ncm|ncm|49| flymodel|flymodel|1| sdmlp_lr0.05|sdmlp|16|0.05"
CHOICES_LTM="ltm_lr0.0001|ltm|10|0.0001"

lane() {
  local c tree model epochs lr order
  for c in $1; do
    IFS='|' read -r tree model epochs lr <<< "$c"
    for order in "3 4 5" "4 5 3" "5 3 4"; do
      if env RUNS="$ROOT/$tree" EPOCHS="$epochs" ${lr:+LR=$lr} bash run_v2.sh continual "$model" "0 1" 1 "$order" \
          >> "$ROOT/$tree.log" 2>&1 < /dev/null; then
        step "done   $tree EPOCHS=$epochs ${lr:+LR=$lr} order $order"
      else
        step "FAILED $tree EPOCHS=$epochs ${lr:+LR=$lr} order $order ($ROOT/$tree.log)"
      fi
    done
  done
}

step "confirmation runs start: code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty')"
lane "$CHOICES_HEADS" & lane "$CHOICES_LTM" & wait
step "confirmation runs finished"
