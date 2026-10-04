#!/bin/bash
# Actions files of the first main CLS/STM cloud round (plan.md section 1b, blocks 4, 5, 7; seeds 1-3 only, Gideon
# 4 Oct 2026). One file = one pod (sky/v2_launch.sh; cluster name = file basename). CLS/STM models are not selected
# (plan.md section 3, decision 5): run_v2.sh defaults only (RL lr 0.1 at batch 16, actor lr 0.01, 12 pre-training
# epochs, 12 epochs per continual phase, 12 epochs at every few-shot N); SAVE_STM=1 on every line.
#   v2c-<model>-p<b|c|d>-s<k>  pairs B-D, e13: pretrain + baseline + 3 orders                       (18 pods, 5 units)
#   v2cf-<model>-pa-s<k>       pair A, e13: pretrain + baseline + 3 orders + few-shot fine 3,4,5 x N (6 pods, 20 units)
#                              (few-shot loads the pair A pre-training of the same model and seed, so it shares the pod)
#   v2e-<model>-pa-s<k>        pair A, LTM=e40: pretrain_e40 + baseline_e40 + 3 orders               (6 pods, 5 units)
# Unit dirs come from run_v2.sh itself (PRINT_UNIT=1). Usage (from the repo root): bash sky/rounds/main1/make_actions.sh
set -eu
cd "$(dirname "$0")/../../.."
OUT=sky/rounds/main1
TREE=runs_v2
ORDERS=("3 4 5" "4 5 3" "5 3 4")
NS="1 4 16 64 400"
line() {  # env, run_v2.sh args (as one string of shell words)
  local env=$1 args=$2 unit
  unit=$(eval "env $env RUNS=$TREE PRINT_UNIT=1 bash run_v2.sh $args")
  [ -n "$unit" ] || { echo "run_v2.sh refused: $env | $args" >&2; exit 1; }
  echo "$TREE|$env|$args|$unit|"
}
pod() {  # file, env, model, pair, seed, with_fewshot
  local f=$OUT/$1.txt env=$2 m=$3 cc=$4 s=$5 fs=$6 o c n
  [ -e "$f" ] && { echo "exists: $f" >&2; exit 1; }
  {
    line "$env" "pretrain $m \"$cc\" $s"
    line "$env" "baseline $m \"$cc\" $s"
    for o in "${ORDERS[@]}"; do line "$env" "continual $m \"$cc\" $s \"$o\""; done
    if [ "$fs" = 1 ]; then for c in 3 4 5; do for n in $NS; do line "$env" "fewshot $m \"$cc\" $s $c $n"; done; done; fi
  } > "$f"
}
for m in rl actor; do
  for s in 1 2 3; do
    pod "v2cf-$m-pa-s$s" "SAVE_STM=1" "$m" "0 1" "$s" 1
    pod "v2c-$m-pb-s$s" "SAVE_STM=1" "$m" "2 3" "$s" 0
    pod "v2c-$m-pc-s$s" "SAVE_STM=1" "$m" "5 6" "$s" 0
    pod "v2c-$m-pd-s$s" "SAVE_STM=1" "$m" "15 16" "$s" 0
    pod "v2e-$m-pa-s$s" "LTM=e40 SAVE_STM=1" "$m" "0 1" "$s" 0
  done
done
ls "$OUT"/*.txt | wc -l
