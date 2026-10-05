#!/bin/bash
# Actions files of the pair A CLS/STM cloud round (plan.md section 1a/1b, blocks 4, 6, 7; seeds 1-3, Gideon 4 Oct 2026).
# One file = one pod (sky/v2_launch.sh; cluster name = file basename). Replaces the never-launched drafts
# sky/rounds/main1/v2cf-*.txt (left untouched) by adding the single-stream units (plan.md section 4.2: pair "0 1",
# fine 3, 4, 5, batch 1) to the same pod, so each model/seed has ONE STM pre-training shared by all its units:
#   v2cfs-<model>-pa-s<k>  pair A, e13: pretrain + stream fine 3,4,5 + 3 continual orders + few-shot fine 3,4,5 x N
#                          + baseline (23 units). Longest units first (pod job: NPROC=3, file order after pre-training).
# Settings (plan.md section 3): CLS/STM continual and few-shot at run_v2.sh defaults (RL lr 0.1, actor lr 0.01, 12 epochs
# per phase / at every N). Single-stream:
#   RL     LR=$RL_LR EPOCHS=$RL_EPOCHS  from the RL single-stream learning-rate sweep (runs_v2_pilot_stm/, selected with the
#          heads' rule on validation only; plan.md section 3 decision 5, "New 2 Oct"). REQUIRED, no default.
#   actor  LR=0.01 EPOCHS=192           the draft's budget and our learning rate (decision 5: CLS/STM not selected; the
#          actor's selection is deferred); equal to run_v2.sh's defaults, written out so the actions file records them.
# Epochs are the draft's: 4,000 minibatch-1 steps each, evaluated every 16th epoch and at the last (run_v2.sh).
# SAVE_STM=1 on every line. Unit dirs come from run_v2.sh itself (PRINT_UNIT=1).
# Usage (repo root, after the sweep's selection): RL_LR=0.01 RL_EPOCHS=192 bash sky/rounds/main2/make_actions.sh
#   OUT=<dir> writes elsewhere (tests). Refuses to overwrite an existing file.
set -eu
cd "$(dirname "$0")/../../.."
OUT=${OUT:-sky/rounds/main2}
RL_LR=${RL_LR:?RL_LR: the RL single-stream learning rate selected from the sweep}
RL_EPOCHS=${RL_EPOCHS:?RL_EPOCHS: the RL single-stream budget (draft epochs of 4,000 steps) selected from the sweep}
[[ "$RL_LR" =~ ^(0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(\.[0-9]+)?)$ ]] || { echo "RL_LR must be a positive decimal: $RL_LR" >&2; exit 2; }
[[ "$RL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "RL_EPOCHS must be a positive whole number: $RL_EPOCHS" >&2; exit 2; }
[ "$RL_EPOCHS" -le 384 ] || { echo "RL_EPOCHS $RL_EPOCHS is above the sweep's cap (384)" >&2; exit 2; }
ACTOR_LR=0.01; ACTOR_EPOCHS=192
mkdir -p "$OUT"
TREE=runs_v2
ORDERS=("3 4 5" "4 5 3" "5 3 4")
NS="1 4 16 64 400"
line() {  # env, run_v2.sh args (as one string of shell words)
  local env=$1 args=$2 unit
  unit=$(eval "env $env RUNS=$TREE PRINT_UNIT=1 bash run_v2.sh $args")
  [ -n "$unit" ] || { echo "run_v2.sh refused: $env | $args" >&2; exit 1; }
  echo "$TREE|$env|$args|$unit|"
}
pod() {  # file, model, pair, seed, stream env
  local f=$OUT/$1.txt m=$2 cc=$3 s=$4 senv=$5 o c n
  [ -e "$f" ] && { echo "exists: $f" >&2; exit 1; }
  {
    line "SAVE_STM=1" "pretrain $m \"$cc\" $s"
    for c in 3 4 5; do line "SAVE_STM=1 $senv" "stream $m \"$cc\" $s $c"; done
    for o in "${ORDERS[@]}"; do line "SAVE_STM=1" "continual $m \"$cc\" $s \"$o\""; done
    for c in 3 4 5; do for n in $NS; do line "SAVE_STM=1" "fewshot $m \"$cc\" $s $c $n"; done; done
    line "SAVE_STM=1" "baseline $m \"$cc\" $s"
  } > "$f"
}
for s in 1 2 3; do
  pod "v2cfs-rl-pa-s$s" rl "0 1" "$s" "LR=$RL_LR EPOCHS=$RL_EPOCHS"
  pod "v2cfs-actor-pa-s$s" actor "0 1" "$s" "LR=$ACTOR_LR EPOCHS=$ACTOR_EPOCHS"
done
ls "$OUT"/v2cfs-*.txt | wc -l
# UNIT_TIMEOUT (one per launch, covers the longest unit): ~2x the slowest stream unit's expected time on a normal host
# (cloud batch 1 measured at ~1 min per 4,000-step epoch or less; plan.md / ledger, 5 Oct estimate).
L=$(( RL_EPOCHS > ACTOR_EPOCHS ? RL_EPOCHS : ACTOR_EPOCHS ))
echo "suggested UNIT_TIMEOUT=$(( (L * 2 + 59) / 60 + 1 ))h (longest stream unit $L epochs)"
