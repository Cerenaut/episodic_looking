#!/bin/bash
# Few-shot / data efficiency (Gideon 2026-09-22: "since we are memorizing"): does the STM need fewer DISTINCT images
# than the alternatives, as a memory should? Distinct from exposure efficiency, which asks how many repetitions it needs.
#
# Design: train on ONE unseen fine class (3) with N training images per coarse class, N in {16, 64, 500}, on the e13 LTM,
# evaluating all four test sets. Every model gets the SAME exposure budget of 12,000 (the LTM's standard phase budget),
# so the axis is matched, unlike the paper's few-shot figures:
#   STM: --training-steps 500 => 500*16/8 = 1,000 exposures per reported epoch, 12 epochs.
#   heads/LTM at batch 16: one epoch = 2N images, so epochs = 12000/(2N) (375 / 94 / 12 for N = 16 / 64 / 500).
# Models: fixed RL STM, A.1 differentiable, LTM-only fine-tuning, linear, NCM, FlyModel, SDMLP.
# Starts after D.15 (run_m3_d15.sh). Results runs_fewshot_*/ and runs_fewshot_heads/; logs runs_local/fewshot/.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/fewshot
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
CLS=3
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

step "waiting for D.15 to finish"
while pgrep -f "^/bin/bash ./run_m3_d15.sh" > /dev/null; do sleep 120; done
step "M3 free; starting few-shot sweep on fine class $CLS"

for N in 16 64 500; do
  EPOCHS=$(( 12000 / (2 * N) ))
  # --- STM variants: 1,000 exposures per reported epoch, 12 epochs = 12,000 exposures ---
  if [ ! -f "$LOG/ref_$N.done" ]; then
    step "fixed RL, N=$N"
    ( $PY cifar_main_stm_training.py --experiment-type few-shot --fine-classes $CLS --coarse-classes 0 1 \
        --max-instances $N --training-steps 500 --epochs 12 --evaluate-steps 800 --eval-bias mean \
        --ltm-checkpoint $E13 --stm-checkpoint $CK/variants/stm_rl_fixed_e13_pt12.pth --run-root runs_fewshot_ref \
        > "$LOG/ref_$N.log" 2>&1 && touch "$LOG/ref_$N.done" && step "fixed RL N=$N done" || step "FAILED: fixed RL N=$N" ) &
  fi
  if [ ! -f "$LOG/a1_$N.done" ]; then
    step "A.1, N=$N"
    ( $PY cifar_main_stm_training.py --experiment-type few-shot --fine-classes $CLS --coarse-classes 0 1 \
        --max-instances $N --training-steps 500 --epochs 12 --evaluate-steps 800 \
        --actor-training differentiable --learning-rate 0.01 \
        --ltm-checkpoint $E13 --stm-checkpoint $CK/variants/stm_a1_diff_lr0.01_e13_pt12.pth --run-root runs_fewshot_a1 \
        > "$LOG/a1_$N.log" 2>&1 && touch "$LOG/a1_$N.done" && step "A.1 N=$N done" || step "FAILED: A.1 N=$N" ) &
  fi
  wait
  # --- heads (3 seeds each) and LTM-only: same 12,000 exposures ---
  for m in linear ncm flymodel sdmlp; do
    [ -f "$LOG/head_${m}_$N.done" ] && continue
    step "head $m, N=$N ($EPOCHS epochs x $((2*N)) images)"
    ( for s in 0 1 2; do
        $PY cifar_main_head_baselines.py --method $m --experiment-type continual --fine-classes $CLS \
          --coarse-classes 0 1 --max-instances $N --epochs $EPOCHS --seed $s --runs-root runs_fewshot_heads || exit 1
      done && touch "$LOG/head_${m}_$N.done" && step "head $m N=$N done" || step "FAILED: head $m N=$N" ) > "$LOG/head_${m}_$N.log" 2>&1 &
  done
  wait
  if [ ! -f "$LOG/ltm_$N.done" ]; then
    step "LTM-only, N=$N"
    ( $PY cifar_main_ltm_fine_tuning.py --experiment-type few-shot --fine-classes $CLS --coarse-classes 0 1 \
        --max-instances $N --learning-rate 0.001 --run-root runs_fewshot_ltm \
        > "$LOG/ltm_$N.log" 2>&1 && touch "$LOG/ltm_$N.done" && step "LTM N=$N done" || step "FAILED: LTM N=$N" ) &
    wait
  fi
done
step "ALL DONE"
