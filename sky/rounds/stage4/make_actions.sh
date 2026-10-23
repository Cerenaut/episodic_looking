#!/bin/bash
# Actions files of the stage4 RL STM round (Dave, 13 Oct 2026): the RL STM with stage-4 LTM features (512, pooled) as
# its encoding (--stm-encoding stage4; the best of the side-quest screens on validation, plan.md section 9), otherwise
# run_v2.sh defaults (lr 0.1 at batch 16, momentum 0.5, 12 pre-training epochs, 12 per continual phase), e13 LTM.
# Tree runs_v2_stage4/, so nothing mixes with the original RL runs in runs_v2/.
#   v2s4-rl-p<a|b|c|d>-s<k>  pretrain + baseline + 3 orders (12 pods, 5 units each)
# Usage (from the repo root): bash sky/rounds/stage4/make_actions.sh
set -eu
cd "$(dirname "$0")/../../.."
OUT=sky/rounds/stage4
TREE=runs_v2_stage4
ENV='SAVE_STM=1 EXTRA="--stm-encoding stage4"'
ORDERS=("3 4 5" "4 5 3" "5 3 4")
line() {  # run_v2.sh args (as one string of shell words)
  local args=$1 unit
  unit=$(eval "env $ENV RUNS=$TREE PRINT_UNIT=1 bash run_v2.sh $args")
  [ -n "$unit" ] || { echo "run_v2.sh refused: $args" >&2; exit 1; }
  echo "$TREE|$ENV|$args|$unit|"
}
for s in 1 2 3; do
  for p in "a:0 1" "b:2 3" "c:5 6" "d:15 16"; do
    f=$OUT/v2s4-rl-p${p%%:*}-s$s.txt cc=${p#*:}
    [ -e "$f" ] && { echo "exists: $f" >&2; exit 1; }
    {
      line "pretrain rl \"$cc\" $s"
      line "baseline rl \"$cc\" $s"
      for o in "${ORDERS[@]}"; do line "continual rl \"$cc\" $s \"$o\""; done
    } > "$f"
  done
done
