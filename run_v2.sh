#!/bin/bash
# v2 launcher (Notes/experiments/plan.md, code item C3): one resumable job per call, any coarse-class pair, with run
# roots that carry setting, model, pair and seed:
#
#   runs_v2/<setting>/<model>/pair<a>_<b>/seed<k>/<unit>/
#
#   setting  pretrain | baseline | continual | stream | fewshot   (+ "_e40" when LTM=e40)
#   model    rl | actor | sup | sup2 | linear | ncm | flymodel | sdmlp | ltm   (sup: CLS/STM trained by supervised
#            learning, without episodes, cifar_main_stm_supervised.py, output layer zero-initialised; sup2: the same
#            with PyTorch's default initialisation, as the RL actor)
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
# Overrides (environment): EPOCHS (per phase / per run), PRETRAIN_EPOCHS, LR, PRETRAIN_LR (STM), TRAINING_STEPS (STM), EVAL_EPOCHS
# (evaluation interval, every model), EVAL_POINTS (log-spaced evaluation, heads and LTM-only), VAL_HOLDOUT, LTM=e13|e40, SAVE_STM=1 (keep STM checkpoints), EXTRA (appended to the script's arguments), PY, DRY=1 (print the
# command only), PRINT_UNIT=1 (print the unit dir only and exit; sky/v2_launch.sh uses it so that the layout lives in
# this file alone), RUNS (default runs_v2), LTM_BN=train|frozen (LTM-only only, below).
# LTM_BN (decided 6 Oct 2026): LTM-only batch normalization while training. train (default) = the draft's training-mode
# BN, every existing run. frozen = eval-mode BN with the BN affine parameters fixed (--bn-mode frozen), used for
# single-stream (minibatch 1). Frozen results live in a separate tree with the same layout, never beside train-mode
# ones: RUNS defaults to runs_v2_bnfrozen, a frozen launch is refused unless RUNS contains "bnfrozen", and an LTM-only
# launch into such a tree is refused unless LTM_BN=frozen. An EXTRA containing --bn-mode is refused (it would bypass this guard). Unit paths inside the tree are unchanged, so metrics_v2.py
# reads either tree as it is.
# Resumable: a finished unit has <unit>/job.done; the script refuses to run into an unfinished directory that already
# holds results (move it to an archive first: results are never overwritten).
set -u
cd "$(dirname "$0")"
PY=${PY:-/Users/gideon/anaconda3/envs/episodic/bin/python}
export PYTORCH_ENABLE_MPS_FALLBACK=1

SETTING=${1:?setting: pretrain|baseline|continual|stream|fewshot}
MODEL=${2:?model: rl|actor|sup|sup2|linear|ncm|flymodel|sdmlp|ltm}
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
LTM_BN=${LTM_BN:-train}
case "$LTM_BN" in
  train) RUNS=${RUNS:-runs_v2} ;;
  frozen) RUNS=${RUNS:-runs_v2_bnfrozen} ;;
  *) echo "unknown LTM_BN: $LTM_BN (train|frozen)" >&2; exit 2 ;;
esac
PAIR="pair$(echo $CC | tr ' ' '_')"
SEEDDIR=$RUNS/$SETTING$SUFFIX/$MODEL/$PAIR/seed$SEED
PRETRAIN_DIR=$RUNS/pretrain$SUFFIX/$MODEL/$PAIR/seed$SEED
STM_CKPT=$PRETRAIN_DIR/stm_pretrain.pth

case "$MODEL" in
  rl|actor) KIND=stm ;;
  sup|sup2) KIND=sup ;;
  linear|ncm|flymodel|sdmlp) KIND=head ;;
  ltm) KIND=ltm ;;
  *) echo "unknown model: $MODEL" >&2; exit 2 ;;
esac
# Keep the two BN modes' results apart (LTM_BN above)
case "$RUNS" in *bnfrozen*) RUNS_FROZEN=1 ;; *) RUNS_FROZEN=0 ;; esac
if [ "$LTM_BN" = frozen ]; then
  [ "$KIND" = ltm ] && [ "$SETTING" != baseline ] || { echo "LTM_BN=frozen applies to LTM-only training units only" >&2; exit 2; }
  [ "$RUNS_FROZEN" = 1 ] || { echo "LTM_BN=frozen needs a RUNS tree whose path contains 'bnfrozen' (got $RUNS)" >&2; exit 2; }
elif [ "$KIND" = ltm ] && [ "$SETTING" != baseline ] && [ "$RUNS_FROZEN" = 1 ]; then
  echo "$RUNS is a frozen-BN tree: set LTM_BN=frozen for LTM-only units there" >&2; exit 2
fi
# The BN mode is set by LTM_BN alone: an EXTRA --bn-mode would bypass the tree guard above
case " ${EXTRA:-} " in
  *--bn-mode*) echo "EXTRA must not contain --bn-mode: set LTM_BN=train|frozen instead (it picks the tree)" >&2; exit 2 ;;
esac

# --- the unit of work -------------------------------------------------------------------------
case "$SETTING" in
  pretrain)
    [ "$KIND" = stm ] || [ "$KIND" = sup ] || { echo "pretrain applies to the STM models only (heads pre-train in their own runs)" >&2; exit 2; }
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
if [ "${PRINT_UNIT:-0}" = 1 ]; then echo "$UNIT_DIR"; exit 0; fi

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
# LTM-only loads in the main process: worker processes are re-spawned every epoch, 7x slower for short epochs, same result
LTM_BASE="cifar_main_ltm_fine_tuning.py --coarse-classes $CC --seed $SEED $SPLIT --ltm-checkpoint $LTM_CKPT --learning-rate ${LR:-0.001} --loader-workers 0"
# Frozen BN only when asked, so a train-mode command is exactly the one every earlier run used
[ "$LTM_BN" = frozen ] && LTM_BASE="$LTM_BASE --bn-mode frozen"
# Supervised CLS/STM (sup): epochs are passes over the training images, as the heads' and LTM-only's; the draft has no
# budget or learning rate for it (selected on validation). Pre-training 12 epochs at PRETRAIN_LR (default 0.01, the
# differentiable actor's), minibatch 16, every epoch evaluated.
SUP_BASE="cifar_main_stm_supervised.py --coarse-classes $CC --seed $SEED $SPLIT --ltm-checkpoint $LTM_CKPT --loader-workers 0"
[ "$MODEL" = sup2 ] && SUP_BASE="$SUP_BASE --sup-init default"
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
  pretrain/sup)
    CMD="$SUP_BASE --learning-rate ${PRETRAIN_LR:-0.01} --experiment-type pretrain --fine-classes 1 2 --epochs ${PRETRAIN_EPOCHS:-12} --stm-checkpoint $STM_CKPT --run-root $UNIT_DIR" ;;
  baseline/sup)
    CMD="$SUP_BASE --experiment-type evaluate --fine-classes 3 4 5 --stm-checkpoint $STM_CKPT --run-root $UNIT_DIR" ;;
  continual/sup)
    CMD="$SUP_BASE --learning-rate ${LR:-0.01} --experiment-type continual --fine-classes $ORDER --epochs ${EPOCHS:-12} --stm-checkpoint $STM_CKPT $(save_stm "$UNIT_DIR/stm") --run-root $UNIT_DIR" ;;
  stream/sup|fewshot/sup)
    echo "$SETTING is not set up for sup yet" >&2; exit 2 ;;
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
# Evaluation interval for the heads and LTM-only (every k-th epoch of a phase and its last); the STM's is above
if [ "$KIND" != stm ] && [ "$SETTING" != baseline ] && [ -n "${EVAL_EPOCHS:-}" ]; then CMD="$CMD --evaluate-epochs $EVAL_EPOCHS"; fi
# Log-spaced evaluation (heads, LTM-only): about EVAL_POINTS epochs per phase, dense early; overrides EVAL_EPOCHS
if [ "$KIND" != stm ] && [ "$SETTING" != baseline ] && [ -n "${EVAL_POINTS:-}" ]; then CMD="$CMD --evaluate-points $EVAL_POINTS"; fi
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
if { [ "$KIND" = stm ] || [ "$KIND" = sup ]; } && [ "$SETTING" != pretrain ] && [ ! -f "$PRETRAIN_DIR/job.done" ]; then
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
