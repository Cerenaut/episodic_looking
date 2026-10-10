#!/bin/bash
# STM input-conditioning side-quest (paper repo Notes/experiments/plan.md section 9): RL STM, pair A (coarse 0,1),
# seed 1, e13 LTM, the final runs' settings (run_v2.sh: lr 0.1, batch 16, 12 epochs, eval sweep, mean bias at
# evaluation, validation split 100/0). One variant = one set of extra arguments; units mirror run_v2.sh's tree under
# runs_cond/: pretrain/<variant>/pair0_1/seed1, baseline/<variant>/... (pre-continual evaluation of fine 3,4,5), and
# continual/<variant>/.../order3_4_5. Resumable: a finished unit has job.done; a started, unfinished unit is refused.
#
# Usage (repo root):
#   bash run_conditioning_screen.sh pretrain <variant>     # STM pre-training, then the pre-continual evaluation
#   bash run_conditioning_screen.sh continual <variant>    # continual order 3,4,5 from that pre-training
#   bash run_conditioning_screen.sh list                   # the variants
# Environment: PY (python), THREADS (torch CPU threads), DRY=1 (print the commands only),
#   CONT_EPOCHS (continual epochs per phase, default 12; other values write to order3_4_5_e<N>).
set -euo pipefail

declare -A VARIANTS=(
  [original]=""
  [centre_mask_first]="--input-conditioning centre --input-conditioning-target mask --mask-key first"
  [centre_both_first]="--input-conditioning centre --input-conditioning-target both --mask-key first"
  [cscale_mask_first]="--input-conditioning centre-scale --input-conditioning-target mask --mask-key first"
  [cscale_both_first]="--input-conditioning centre-scale --input-conditioning-target both --mask-key first"
  [centre_mask_current_bw0]="--input-conditioning centre --input-conditioning-target mask --mask-key current --mask-key-bias-weight 0"
  [centre_both_current_bw0]="--input-conditioning centre --input-conditioning-target both --mask-key current --mask-key-bias-weight 0"
  [cscale_mask_current_bw0]="--input-conditioning centre-scale --input-conditioning-target mask --mask-key current --mask-key-bias-weight 0"
  [cscale_both_current_bw0]="--input-conditioning centre-scale --input-conditioning-target both --mask-key current --mask-key-bias-weight 0"
  [centre_mask_history_bw0]="--input-conditioning centre --input-conditioning-target mask --mask-key history --mask-key-bias-weight 0"
  [centre_both_history_bw0]="--input-conditioning centre --input-conditioning-target both --mask-key history --mask-key-bias-weight 0"
  [cscale_mask_history_bw0]="--input-conditioning centre-scale --input-conditioning-target mask --mask-key history --mask-key-bias-weight 0"
  [cscale_both_history_bw0]="--input-conditioning centre-scale --input-conditioning-target both --mask-key history --mask-key-bias-weight 0"
  [layernorm_mask_first]="--input-conditioning layernorm --input-conditioning-target mask --mask-key first"
  # logits zeroed in the mask key (lw0; key keeps its size). With bias and logits both out of the key, centre and
  # centre-scale give the same mask (one part left), so only centre variants.
  [centre_mask_current_bw0_lw0]="--input-conditioning centre --input-conditioning-target mask --mask-key current --mask-key-bias-weight 0 --mask-key-logit-weight 0"
  [centre_mask_first_lw0]="--input-conditioning centre --input-conditioning-target mask --mask-key first --mask-key-logit-weight 0"
  [centre_mask_history_bw0_lw0]="--input-conditioning centre --input-conditioning-target mask --mask-key history --mask-key-bias-weight 0 --mask-key-logit-weight 0"
  # STM architecture ablations on the original input (Dave, 11 Oct). Both apply to actor and critic.
  [original_noaffine_nooutbias]="--stm-no-layer-norm-affine --stm-no-output-bias"
  # LTM's final-stage pooled features (512) in place of the bias-stage encoding, in model input and mask key alike
  [original_stage4]="--stm-encoding stage4"
  # both of the above (2x2 with original, original_noaffine_nooutbias, original_stage4)
  [original_stage4_noaffine_nooutbias]="--stm-encoding stage4 --stm-no-layer-norm-affine --stm-no-output-bias"
)

SETTING=${1:?setting: pretrain|continual|list}
if [ "$SETTING" = list ]; then for v in "${!VARIANTS[@]}"; do echo "$v: ${VARIANTS[$v]}"; done | sort; exit 0; fi
VARIANT=${2:?variant (see: list)}
[ -n "${VARIANTS[$VARIANT]+x}" ] || { echo "unknown variant: $VARIANT" >&2; exit 2; }
EXTRA=${VARIANTS[$VARIANT]}

PY=${PY:-../.venv/bin/python}
LTM_CKPT=../cifar_100_pretrain/cifar_100_subclasses_12_e13.pth
[ -f "$LTM_CKPT" ] || { echo "missing LTM checkpoint: $LTM_CKPT" >&2; exit 2; }
RUNS=runs_cond
PRETRAIN_DIR=$RUNS/pretrain/$VARIANT/pair0_1/seed1
BASELINE_DIR=$RUNS/baseline/$VARIANT/pair0_1/seed1
CONT_EPOCHS=${CONT_EPOCHS:-12}
CONTINUAL_DIR=$RUNS/continual/$VARIANT/pair0_1/seed1/order3_4_5
[ "$CONT_EPOCHS" = 12 ] || CONTINUAL_DIR=${CONTINUAL_DIR}_e$CONT_EPOCHS
STM_CKPT=$PRETRAIN_DIR/stm_pretrain.pth
STM_BASE="cifar_main_stm_training.py --coarse-classes 0 1 --seed 1 --val-holdout 100 --split-seed 0 --eval-sweep --eval-batch-size 16 --ltm-checkpoint $LTM_CKPT --eval-bias mean $EXTRA"

run_unit() {  # <unit dir> <command>
  local UNIT_DIR=$1 CMD=$2
  if [ -f "$UNIT_DIR/job.done" ]; then echo "done already: $UNIT_DIR"; return 0; fi
  if [ -d "$UNIT_DIR" ] && [ -n "$(ls -A "$UNIT_DIR")" ]; then
    echo "refusing to run into a started, unfinished unit: $UNIT_DIR (archive it first)" >&2; return 1
  fi
  if [ "${DRY:-0}" = 1 ]; then echo "$PY $CMD"; return 0; fi
  mkdir -p "$UNIT_DIR"
  { git rev-parse HEAD; git status --porcelain --untracked-files=no | sed 's/^/dirty: /'; } > "$UNIT_DIR/code_commit.txt"
  echo "[$(date '+%F %T')] start ($VARIANT): $PY $CMD" | tee -a "$UNIT_DIR/job.log"
  local rc=0
  OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} $PY $CMD >> "$UNIT_DIR/job.log" 2>&1 || rc=$?
  if [ $rc = 0 ]; then
    touch "$UNIT_DIR/job.done"; echo "[$(date '+%F %T')] done: $UNIT_DIR"
  else
    echo "[$(date '+%F %T')] FAILED ($rc): $UNIT_DIR; log $UNIT_DIR/job.log" >&2; return $rc
  fi
}

case "$SETTING" in
  pretrain)
    run_unit "$PRETRAIN_DIR" "$STM_BASE --learning-rate 0.1 --experiment-type pretrain --fine-classes 1 2 --epochs 12 --stm-checkpoint $STM_CKPT --run-root $PRETRAIN_DIR"
    run_unit "$BASELINE_DIR" "$STM_BASE --experiment-type evaluate --fine-classes 3 4 5 --stm-checkpoint $STM_CKPT --run-root $BASELINE_DIR" ;;
  continual)
    [ -f "$PRETRAIN_DIR/job.done" ] || { echo "pre-training not finished: $PRETRAIN_DIR" >&2; exit 2; }
    run_unit "$CONTINUAL_DIR" "$STM_BASE --learning-rate 0.1 --experiment-type continual --fine-classes 3 4 5 --epochs $CONT_EPOCHS --stm-checkpoint $STM_CKPT --stm-checkpoint-out $CONTINUAL_DIR/stm --run-root $CONTINUAL_DIR" ;;
  *) echo "unknown setting: $SETTING" >&2; exit 2 ;;
esac
