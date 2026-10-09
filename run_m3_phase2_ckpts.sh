#!/bin/bash
# Phase 2 run 2 (research_plan.md): one continual order (3,4,5) of the fixed RL reference and of A.1 lr 0.01 on e13, saving the
# STM after each phase (--stm-checkpoint-out) for the inspectability analyses A3/A4/B1-B3 on post-continual models.
# Same settings as the reference runs (mean bias, 800 eval steps), unseeded. Two processes in parallel on the M3 (~6 h).
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/phase2_ckpts
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
if [ ! -f "$LOG/ref.done" ]; then
  step "fixed RL order 3,4,5 with per-phase checkpoints"
  ( $PY cifar_main_stm_training.py --experiment-type continual --fine-classes 3 4 5 --coarse-classes 0 1 --eval-bias mean --evaluate-steps 800 \
      --ltm-checkpoint $E13 --stm-checkpoint $CK/variants/stm_rl_fixed_e13_pt12.pth --stm-checkpoint-out $CK/variants/stm_rl_fixed_e13_pt12_c345 \
      --run-root runs_phase2_ref > "$LOG/ref.log" 2>&1 && touch "$LOG/ref.done" && step "fixed RL finished" || step "FAILED: fixed RL" ) &
fi
if [ ! -f "$LOG/a1.done" ]; then
  step "A.1 lr 0.01 order 3,4,5 with per-phase checkpoints"
  ( $PY cifar_main_stm_training.py --experiment-type continual --fine-classes 3 4 5 --coarse-classes 0 1 --actor-training differentiable --learning-rate 0.01 --evaluate-steps 800 \
      --ltm-checkpoint $E13 --stm-checkpoint $CK/variants/stm_a1_diff_lr0.01_e13_pt12.pth --stm-checkpoint-out $CK/variants/stm_a1_diff_lr0.01_e13_pt12_c345 \
      --run-root runs_phase2_a1 > "$LOG/a1.log" 2>&1 && touch "$LOG/a1.done" && step "A.1 finished" || step "FAILED: A.1" ) &
fi
wait
step "ALL DONE"
