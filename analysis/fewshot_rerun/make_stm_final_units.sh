#!/bin/bash
# Supervised STM head (--method stm) few-shot final runs (Dave, 14 Oct 2026; plan.md section 3, decision 10), one per line
# (environment|run_v2.sh arguments): tree runs_v2 beside the other heads; lr 0.01, pre-training lr 0.01, EVAL_POINTS=96;
# budgets from its selection (metrics_v2.py --plateau, all plateaued): N = 1/4/16/64/400 = 1/8/42/6/1 head epochs;
# seeds 1-10 at N = 1 and 4, seeds 1-5 at N = 16, 64, 400.
# Usage (repo root): bash analysis/fewshot_rerun/make_stm_final_units.sh > runs_local/fewshot_rodney_20261014/stm_final_units.txt
set -eu
STMX='EXTRA="--pretrain-lr 0.01 --device cpu"'
ep() { case $1 in 1) echo 1;; 4) echo 8;; 16) echo 42;; 64) echo 6;; 400) echo 1;; esac; }
for n in 1 4 16 64 400; do
  seeds="1 2 3 4 5"; [ $n -le 4 ] && seeds="1 2 3 4 5 6 7 8 9 10"
  for s in $seeds; do for c in 3 4 5; do
    echo "RUNS=runs_v2 LR=0.01 EPOCHS=$(ep $n) EVAL_POINTS=96 $STMX|fewshot stm \"0 1\" $s $c $n"
  done; done
done
