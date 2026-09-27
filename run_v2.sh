#!/bin/bash
# v2 launcher (Notes/experiments/plan.md, code item C3): one resumable job per call, any coarse-class pair, with run
# roots that carry setting, model, pair and seed:
#
#   runs_v2/<setting>/<model>/pair<a>_<b>/seed<k>/<unit>/
#
#   setting  pretrain | baseline | continual | stream | fewshot   (+ "_e40" when LTM=e40)
#   model    rl | actor | linear | ncm | flymodel | sdmlp | ltm
#   unit     continual: order<a>_<b>_<c>; stream: fine<c>; fewshot: fine<c>_n<N>; pretrain/baseline: none
#            (LTM-only baseline: runs_v2/baseline/ltm/pair<a>_<b>/, no seed)
#
# Usage: bash run_v2.sh <setting> <model> "<coarse pair>" <seed> [order "3 4 5" | fine class] [N]
#   bash run_v2.sh pretrain  rl     "2 3" 1                 # STM pre-training on fine 1,2 (needed by the STM jobs)
#   bash run_v2.sh baseline  rl     "2 3" 1                 # pre-continual evaluation of that checkpoint (BWT0, FWT)
#   bash run_v2.sh baseline  ltm    "2 3" 0                 # the same for the LTM itself (one per pair)
#   bash run_v2.sh continual rl     "2 3" 1 "4 5 3"
#   bash run_v2.sh stream    ncm    "0 1" 1 3
#   bash run_v2.sh fewshot   ltm    "0 1" 1 3 16
#
# Protocol defaults (plan.md section 3): validation split --val-holdout 100 --split-seed 0 everywhere; the STM
# evaluates by one pass over every test and validation image (--eval-sweep) at the policy mean, 16 environments;
# budgets are the draft's, in each model's own epochs, until the pilot sets them:
#   STM   continual 12/phase, stream 192 epochs of 4,000 steps at batch 1 (evaluated every 16 epochs), few-shot 12
#         at every N;
#         pre-training 12 epochs at batch 16 (stream jobs use the same checkpoint).
#   heads continual 12/phase, stream 12, few-shot floor(6250/N); pre-trained 12 epochs on all of fine 1,2 (C6).
#   LTM   continual 12/phase, stream 12, few-shot floor(6250/N).
# The seed also selects the few-shot images, the same ones in every model for a given seed (C9). Use the same seed
# numbers for every model.
#
# Overrides (environment): EPOCHS (per phase / per run), PRETRAIN_EPOCHS, LR, PRETRAIN_LR (STM), TRAINING_STEPS (STM), EVAL_EPOCHS (STM
# evaluation interval), VAL_HOLDOUT, LTM=e13|e40, SAVE_STM=1 (keep STM checkpoints), EXTRA (appended to the script's arguments), PY, DRY=1 (print the
# command only), RUNS (default runs_v2).
# Resumable: a finished unit has <unit>/job.done; the script refuses to run into an unfinished directory that already
# holds results (move it to an archive first: results are never overwritten).
set -u
cd "$(dirname "$0")"
PY=${PY:-/Users/gideon/anaconda3/envs/episodic/bin/python}
export PYTORCH_ENABLE_MPS_FALLBACK=1

SETTING=${1:?setting: pretrain|baseline|continual|stream|fewshot}
MODEL=${2:?model: rl|actor|linear|ncm|flymodel|sdmlp|ltm}
CC=${3:?coarse pair, e.g. \"0 1\"}
SEED=${4:?seed}
ARG5=${5:-}
N=${6:-}

CK=../cifar_100_pretrain
LTM=${LTM:-e13}
case "$LTM" in
  e13) LTM_CKPT=$CK/cifar_100_subclasses_12_e13.pth; SUFFIX="" ;;
  e40) LTM_CKPT=$CK/cifar_100_subclasses_12_e40.pth; SUFFIX="_e40" ;;
  *) echo "unknown LTM: $LTM" >&2; exit 2 ;;
esac
[ -f "$LTM_CKPT" ] || { echo "missing LTM checkpoint: $LTM_CKPT" >&2; exit 2; }

VAL_HOLDOUT=${VAL_HOLDOUT:-100}
SPLIT="--val-holdout $VAL_HOLDOUT --split-seed 0"
RUNS=${RUNS:-runs_v2}
PAIR="pair$(echo $CC | tr ' ' '_')"
SEEDDIR=$RUNS/$SETTING$SUFFIX/$MODEL/$PAIR/seed$SEED
PRETRAIN_DIR=$RUNS/pretrain$SUFFIX/$MODEL/$PAIR/seed$SEED
STM_CKPT=$PRETRAIN_DIR/stm_pretrain.pth

case "$MODEL" in
  rl|actor) KIND=stm ;;
  linear|ncm|flymodel|sdmlp) KIND=head ;;
  ltm) KIND=ltm ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac

# --- the unit of work -------------------------------------------------------------------------
case "$SETTING" in
  pretrain)
    [ "$KIND" = stm ] || { echo "pretrain applies to the STM models only (heads pre-train in their own runs)" >&2; exit 2; }
    UNIT_DIR=$SEEDDIR ;;
  baseline)
    # LTM-only: the frozen LTM itself, deterministic, so one per pair (the seed is ignored)
    [ "$KIND" = head ] && { echo "heads write their baseline (results_pretrain*.txt) in every run" >&2; exit 2; }
    UNIT_DIR=$SEEDDIR; [ "$KIND" = ltm ] && UNIT_DIR=$RUNS/baseline$SUFFIX/ltm/$PAIR ;;
  continual)
    ORDER=${ARG5:-3 4 5}; UNIT_DIR=$SEEDDIR/order$(echo $ORDER | tr ' ' '_') ;;
  stream)
    CLS=${ARG5:?fine class}; UNIT_DIR=$SEEDDIR/fine$CLS ;;
  fewshot)
    CLS=${ARG5:?fine class}; : "${N:?N images per coarse class}"; UNIT_DIR=$SEEDDIR/fine${CLS}_n$N ;;
  *) echo "unknown setting: $SETTING" >&2; exit 2 ;;
esac

# --- the command -------------------------------------------------------------------------------
STM_BASE="cifar_main_stm_training.py --coarse-classes $CC --seed $SEED $SPLIT --eval-sweep --eval-batch-size 16 --ltm-checkpoint $LTM_CKPT"
# STM learning rates: RL 0.1 at batch 16 and 0.01 at batch 1; differentiable actor 0.01. LR overrides the training
# rate only; pre-training keeps its prior rate (plan.md section 3, decision 3) unless PRETRAIN_LR is set.
if [ "$MODEL" = actor ]; then
  STM_BASE="$STM_BASE --actor-training differentiable"
  STM_LR="--learning-rate ${LR:-0.01}"; STM_PRETRAIN_LR="--learning-rate ${PRETRAIN_LR:-0.01}"; STM_STREAM_LR=$STM_LR
else
  STM_BASE="$STM_BASE --eval-bias mean"
  STM_LR="--learning-rate ${LR:-0.1}"; STM_PRETRAIN_LR="--learning-rate ${PRETRAIN_LR:-0.1}"
  STM_STREAM_LR="--learning-rate ${LR:-0.01}"
fi
HEAD_BASE="cifar_main_head_baselines.py --method $MODEL --coarse-classes $CC --seed $SEED $SPLIT --checkpoint $LTM_CKPT --run-path $UNIT_DIR"
LTM_BASE="cifar_main_ltm_fine_tuning.py --coarse-classes $CC --seed $SEED $SPLIT --ltm-checkpoint $LTM_CKPT --learning-rate ${LR:-0.001}"
# STM checkpoints after training (per phase in continual) only with SAVE_STM=1: ~44 MB each.
save_stm() { [ "${SAVE_STM:-0}" = 1 ] && echo "--stm-checkpoint-out $1"; }

case "$SETTING/$KIND" in
  pretrain/stm)
    CMD="$STM_BASE $STM_PRETRAIN_LR --experiment-type pretrain --fine-classes 1 2 --epochs ${PRETRAIN_EPOCHS:-12} --stm-checkpoint $STM_CKPT --run-root $UNIT_DIR" ;;
  baseline/stm)
    CMD="$STM_BASE --experiment-type evaluate --fine-classes 3 4 5 --stm-checkpoint $STM_CKPT --run-root $UNIT_DIR" ;;
  continual/stm)
    CMD="$STM_BASE $STM_LR --experiment-type continual --fine-classes $ORDER --epochs ${EPOCHS:-12} --stm-checkpoint $STM_CKPT $(save_stm "$UNIT_DIR/stm") --run-root $UNIT_DIR" ;;
  stream/stm)
    # Single-stream: batch 1 from the batch-16 pre-trained checkpoint, lr 0.01 (0.1 diverges at batch 1). Epochs in the
    # draft's units, 4,000 steps each (192 at the draft's budget), evaluated every 16th epoch and at the last.
    CMD="$STM_BASE $STM_STREAM_LR --experiment-type few-shot --fine-classes $CLS --batch-size 1 --epochs ${EPOCHS:-192} --training-steps ${TRAINING_STEPS:-4000} --evaluate-epochs ${EVAL_EPOCHS:-16} --stm-checkpoint $STM_CKPT $(save_stm "$UNIT_DIR/stm_final.pth") --run-root $UNIT_DIR" ;;
  fewshot/stm)
    CMD="$STM_BASE $STM_LR --experiment-type few-shot --fine-classes $CLS --max-instances $N --epochs ${EPOCHS:-12} --stm-checkpoint $STM_CKPT $(save_stm "$UNIT_DIR/stm_final.pth") --run-root $UNIT_DIR" ;;
  baseline/ltm)
    CMD="eval_pretrained_baseline.py --checkpoint $LTM_CKPT --coarse-classes $CC $SPLIT --out $UNIT_DIR/results_evaluate.txt" ;;
  continual/head)
    CMD="$HEAD_BASE --experiment-type continual --fine-classes $ORDER" ;;
  stream/head)
    CMD="$HEAD_BASE --experiment-type streaming --fine-classes $CLS" ;;
  fewshot/head)
    CMD="$HEAD_BASE --experiment-type continual --fine-classes $CLS --max-instances $N --pretrain-all-instances" ;;
  continual/ltm)
    CMD="$LTM_BASE --experiment-type continual --fine-classes $ORDER --run-root $UNIT_DIR" ;;
  stream/ltm)
    CMD="$LTM_BASE --experiment-type streaming --fine-classes $CLS --batch-size 1 --run-root $UNIT_DIR" ;;
  fewshot/ltm)
    CMD="$LTM_BASE --experiment-type few-shot --fine-classes $CLS --max-instances $N --run-root $UNIT_DIR" ;;
esac
# Epoch and learning-rate overrides for the heads and LTM-only (the STM's are in the commands above)
if [ "$KIND" = head ]; then
  [ -n "${EPOCHS:-}" ] && CMD="$CMD --epochs $EPOCHS"
  [ -n "${PRETRAIN_EPOCHS:-}" ] && CMD="$CMD --pretrain-epochs $PRETRAIN_EPOCHS"
  [ -n "${LR:-}" ] && CMD="$CMD --lr $LR"
fi
if [ "$KIND" = ltm ] && [ "$SETTING" != baseline ] && [ -n "${EPOCHS:-}" ]; then CMD="$CMD --epochs $EPOCHS"; fi
if [ "$KIND" = stm ] && [ "$SETTING" != stream ]; then  # stream sets both in its command
  [ -n "${TRAINING_STEPS:-}" ] && CMD="$CMD --training-steps $TRAINING_STEPS"
  [ -n "${EVAL_EPOCHS:-}" ] && CMD="$CMD --evaluate-epochs $EVAL_EPOCHS"
fi
CMD="$CMD ${EXTRA:-}"

if [ "${DRY:-0}" = 1 ]; then echo "$PY $CMD"; exit 0; fi

# --- run ---------------------------------------------------------------------------------------
if [ -f "$UNIT_DIR/job.done" ]; then echo "done already: $UNIT_DIR"; exit 0; fi
if [ -d "$UNIT_DIR" ] && [ -n "$(find "$UNIT_DIR" -name 'results_*.txt' -print -quit)" ]; then
  echo "unfinished results in $UNIT_DIR: move it to an archive before re-running" >&2; exit 3
fi
if [ "$KIND" = stm ] && [ "$SETTING" != pretrain ] && [ ! -f "$PRETRAIN_DIR/job.done" ]; then
  echo "no finished pre-training for this model, pair and seed: run '${SUFFIX:+LTM=$LTM }bash run_v2.sh pretrain $MODEL \"$CC\" $SEED' first ($PRETRAIN_DIR)" >&2
  exit 3
fi
mkdir -p "$UNIT_DIR"
# One launch per unit at a time: mkdir is atomic. A lock left by a killed launch must be removed by hand, after
# checking that no process is still running the unit.
if ! mkdir "$UNIT_DIR/.lock" 2>/dev/null; then
  echo "$UNIT_DIR is locked by another launch (or a killed one: check, then remove $UNIT_DIR/.lock)" >&2; exit 3
fi
trap 'rmdir "$UNIT_DIR/.lock" 2>/dev/null' EXIT
# job.log and code_commit.txt are appended to, so a failed attempt's log is kept when the unit is re-run.
echo "[$(date '+%F %T')] start: $PY $CMD" | tee -a "$UNIT_DIR/job.log"
{ echo "[$(date '+%F %T')]"; git rev-parse HEAD; git status --porcelain -- '*.py'; } >> "$UNIT_DIR/code_commit.txt" 2>/dev/null
# shellcheck disable=SC2086  # CMD is word-split on purpose (bash, not zsh)
$PY $CMD >> "$UNIT_DIR/job.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then
  touch "$UNIT_DIR/job.done"; echo "[$(date '+%F %T')] done: $UNIT_DIR"
else
  echo "[$(date '+%F %T')] FAILED ($rc): $UNIT_DIR; log $UNIT_DIR/job.log" >&2
fi
exit $rc
