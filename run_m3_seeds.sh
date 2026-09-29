#!/bin/bash
# 5-seed round on the M3 (Gideon, 2026-09-21: proceed; M3 only). For seed in 1..5 and variant in {fixed RL reference, A.1 lr 0.01}:
# seeded STM pre-training 12 epochs on e13, then the 3 continual orders in parallel with the same seed. Jobs interleave the two
# variants so both have equal n at any time. Fixed code, --eval-bias mean (RL), --evaluate-steps 800, as the unseeded repeats.
# Results runs_seed/<variant>_s<k>/; checkpoints ../cifar_100_pretrain/variants/stm_<variant>_e13_pt12_seed<k>.pth;
# logs and .done markers runs_local/seeds/. Resumable: re-run the script and it skips finished stages.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/seeds
mkdir -p "$LOG" runs_seed
E13=$CK/cifar_100_subclasses_12_e13.pth
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }

run_job() {  # $1 = variant tag (ref|a1), $2 = seed
  local v=$1 seed=$2 stm root flags
  if [ "$v" = ref ]; then
    stm=$CK/variants/stm_rl_fixed_e13_pt12_seed$seed.pth; flags="--eval-bias mean"
  else
    stm=$CK/variants/stm_a1_diff_lr0.01_e13_pt12_seed$seed.pth; flags="--actor-training differentiable --learning-rate 0.01"
  fi
  root=runs_seed/${v}_s$seed
  if [ ! -f "$LOG/${v}_s${seed}_pretrain.done" ]; then
    step "$v seed $seed: pre-training 12 epochs"
    $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes 0 1 --epochs 12 --evaluate-steps 800 --seed $seed $flags \
        --ltm-checkpoint $E13 --stm-checkpoint $stm --run-root $root > "$LOG/${v}_s${seed}_pretrain.log" 2>&1 \
        && touch "$LOG/${v}_s${seed}_pretrain.done" || { step "FAILED: $v seed $seed pre-training"; return 1; }
  fi
  for order in "3 4 5" "4 5 3" "5 3 4"; do
    local tag; tag=$(echo "$order" | tr -d ' ')
    [ -f "$LOG/${v}_s${seed}_continual_$tag.done" ] && continue
    step "$v seed $seed: continual order $order (parallel)"
    ( $PY cifar_main_stm_training.py --experiment-type continual --fine-classes $order --coarse-classes 0 1 --evaluate-steps 800 --seed $seed $flags \
        --ltm-checkpoint $E13 --stm-checkpoint $stm --run-root $root > "$LOG/${v}_s${seed}_continual_$tag.log" 2>&1 \
        && touch "$LOG/${v}_s${seed}_continual_$tag.done" && step "$v seed $seed order $order finished" || step "FAILED: $v seed $seed order $order" ) &
  done
  wait
}

for seed in 1 2 3 4 5; do
  for v in ref a1; do
    run_job $v $seed
  done
done
step "ALL DONE"
