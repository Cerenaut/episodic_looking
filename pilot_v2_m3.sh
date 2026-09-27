#!/bin/bash
# M3 part of the v2 pilot (Notes/experiments/plan.md, section 3; phase 1b): the heads and LTM-only on pair A, seed 1,
# run to 8x the draft's budget, with a short learning-rate sweep, evaluating on the validation split. Budgets are
# then chosen with metrics_v2.py --part val --plateau (smallest epoch within 0.01 of the maximum, fine 3-5 only).
#
#   continual  3 orders, 96 epochs per phase (draft 12), every epoch evaluated
#   stream     fine 3, minibatch 1, 96 epochs (draft 12), every epoch evaluated
#   few-shot   fine 3, N in {1, 4, 16, 64, 400}, 8 x floor(6250/N) epochs, evaluated ~96 times (every ceil(E/96))
#
# Learning rates (plan: "a short sweep, e.g. LTM-only 1e-3/1e-4, linear/SDMLP 3 values each"; NCM and FlyModel not
# swept): linear 0.0003 / 0.001 / 0.003 (around its validated 0.001), SDMLP 0.015 / 0.05 / 0.15 (around 0.05),
# LTM-only 0.001 / 0.0001. Pre-training keeps its prior settings.
#
# One tree per model and learning rate: runs_v2_pilot/<model>[_lr<x>]/ in run_v2.sh's layout. Two lanes in parallel:
# the heads (CPU) and LTM-only (MPS). Resumable: re-running skips finished units (run_v2.sh). Progress and failures in
# runs_v2_pilot/pilot.log. Usage: bash pilot_v2_m3.sh
set -u
cd "$(dirname "$0")"
LOG=runs_v2_pilot/pilot.log
mkdir -p runs_v2_pilot
step() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
CC="0 1"; SEED=1
ORDERS=("3 4 5" "4 5 3" "5 3 4")
NS=(1 4 16 64 400)

unit() {  # tree, then run_v2.sh arguments; environment overrides are passed by the caller
  local tree=$1; shift
  if RUNS=runs_v2_pilot/$tree bash run_v2.sh "$@" >> "runs_v2_pilot/$tree.log" 2>&1; then
    step "done   $tree: $*"
  else
    step "FAILED $tree: $* (runs_v2_pilot/$tree.log)"
  fi
}

settings() {  # tree, model, [LR]
  local tree=$1 model=$2 lr=${3:-}
  for n in "${NS[@]}"; do
    local e=$(( 8 * (6250 / n) )); local k=$(( (e + 95) / 96 ))
    LR=$lr EPOCHS=$e EVAL_EPOCHS=$k unit "$tree" fewshot "$model" "$CC" $SEED 3 "$n"
  done
  LR=$lr EPOCHS=96 unit "$tree" stream "$model" "$CC" $SEED 3
  for o in "${ORDERS[@]}"; do LR=$lr EPOCHS=96 unit "$tree" continual "$model" "$CC" $SEED "$o"; done
}

heads_lane() {
  settings ncm ncm
  settings flymodel flymodel
  for lr in 0.0003 0.001 0.003; do settings "linear_lr$lr" linear "$lr"; done
  for lr in 0.015 0.05 0.15; do settings "sdmlp_lr$lr" sdmlp "$lr"; done
  step "heads lane finished"
}

ltm_lane() {
  for lr in 0.001 0.0001; do
    unit "ltm_lr$lr" baseline ltm "$CC" 0
    settings "ltm_lr$lr" ltm "$lr"
  done
  step "LTM lane finished"
}

step "pilot start: code $(git rev-parse --short HEAD)$(git diff --quiet -- '*.py' '*.sh' || echo ' +dirty')"
heads_lane & ltm_lane & wait
step "pilot finished: $(grep -c '] done ' "$LOG") units done, $(grep -c '] FAILED ' "$LOG") failed (cumulative over launches)"
