#!/bin/bash
# Item 20 on the M3 (fixed RL code, --eval-bias mean): STM pre-training 12 epochs on LTM epochs 9, 20, 30 (e5 on Walter).
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/sweep
mkdir -p "$LOG"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
for e in 9 20 30; do
  [ -f "$LOG/sweep_e$e.done" ] && continue
  step "item 20: STM pre-training 12 epochs on LTM e$e"
  $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes 0 1 --epochs 12 --eval-bias mean \
     --ltm-checkpoint $CK/cifar_100_subclasses_12_e$e.pth --stm-checkpoint $CK/variants/stm_rl_fixed_e${e}_pt12.pth \
     --run-root runs_p0_sweep/e$e > "$LOG/sweep_e$e.log" 2>&1 && touch "$LOG/sweep_e$e.done" || step "FAILED: sweep e$e"
done
step "ALL DONE"
