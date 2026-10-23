#!/bin/bash
# Rodney (CPU) units of the few-shot rerun (Dave, 14 Oct 2026; plan.md section 3, decision 10), one per line:
#   environment|run_v2.sh arguments
# 1. Supervised STM head (--method stm) budget selection, round 1: tree runs_v2_pilot_fewshot/stm_lr0.01, lr 0.01 (its
#    continual selection), pre-training lr 0.01 as its finals; 8x the heads' starting budget floor(6250/N); seeds 1-3 at
#    N = 1, 4 and seed 1 above (the heads' selection); EVAL_POINTS=96 (log-spaced). Later rounds from metrics_v2.py --plateau.
# 2. Seeds 6-10 at N = 1 and 4 (10 seeds there) for LTM-only fine-tuning and the four heads at their selected
#    budgets and learning rates (final_ltm_v2.sh and final_heads_v2.sh, copied), into runs_v2/ beside seeds 1-5.
# Usage (repo root): bash analysis/fewshot_rerun/make_units.sh > runs_local/fewshot_rodney_20261014/units.txt
set -eu
STMX='EXTRA="--pretrain-lr 0.01 --device cpu"'
for n in 1 4 16 64 400; do
  seeds="1"; [ $n -le 4 ] && seeds="1 2 3"
  for s in $seeds; do for c in 3 4 5; do
    echo "RUNS=runs_v2_pilot_fewshot/stm_lr0.01 LR=0.01 EPOCHS=$(( 8 * (6250 / n) )) EVAL_POINTS=96 $STMX|fewshot stm \"0 1\" $s $c $n"
  done; done
done
ltm_ep() { case $1 in 1) echo 25246 ;; 4) echo 24 ;; esac; }
for n in 1 4; do for s in 6 7 8 9 10; do for c in 3 4 5; do
  echo "RUNS=runs_v2 EPOCHS=$(ltm_ep $n) LR=0.001 EVAL_POINTS=96|fewshot ltm \"0 1\" $s $c $n"
done; done; done
head_ep() { case $1:$2 in linear:1) echo 1;; linear:4) echo 589;; ncm:1) echo 1;; ncm:4) echo 702;; flymodel:1) echo 1;; flymodel:4) echo 1;; sdmlp:1) echo 1;; sdmlp:4) echo 702;; esac; }
head_lr() { case $1 in linear) echo " LR=0.001";; sdmlp) echo " LR=0.05";; *) echo "";; esac; }
for m in linear ncm flymodel sdmlp; do for n in 1 4; do for s in 6 7 8 9 10; do for c in 3 4 5; do
  echo "RUNS=runs_v2 EPOCHS=$(head_ep $m $n)$(head_lr $m) EVAL_POINTS=96|fewshot $m \"0 1\" $s $c $n"
done; done; done; done
