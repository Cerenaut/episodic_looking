#!/bin/bash
# Actions file of the RL STM few-shot budget selection (Dave, 14 Oct 2026; plan.md section 3, decision 10): pair A,
# seed 1, fine 3, 4, 5 x N in 1 4 16 64 400, 24 epochs (2x the draft's 12; Dave's bound from the 24-epoch continual
# round), validation scored every epoch; the budget per N is then chosen by metrics_v2.py --plateau (four-set curve,
# heads' rule). lr 0.1 (run_v2.sh default; tree name rl_lr0.1 for the selection tool). One pod, one STM pre-training
# shared by all 15 units. Tree runs_v2_pilot_fewshot/rl_lr0.1.
#   v2fsel-rl-pa-s1  pretrain + 15 few-shot units
# Usage (from the repo root): bash sky/rounds/fsel/make_actions.sh
set -eu
cd "$(dirname "$0")/../../.."
OUT=sky/rounds/fsel
TREE=runs_v2_pilot_fewshot/rl_lr0.1
line() {  # env, run_v2.sh args (as one string of shell words)
  local env=$1 args=$2 unit
  unit=$(eval "env $env RUNS=$TREE PRINT_UNIT=1 bash run_v2.sh $args")
  [ -n "$unit" ] || { echo "run_v2.sh refused: $args" >&2; exit 1; }
  echo "$TREE|$env|$args|$unit|"
}
f=$OUT/v2fsel-rl-pa-s1.txt
[ -e "$f" ] && { echo "exists: $f" >&2; exit 1; }
{
  line "LR=0.1" "pretrain rl \"0 1\" 1"
  # largest N first, so the longest units start first after pre-training (all units take the same epochs)
  for n in 400 64 16 4 1; do for c in 3 4 5; do line "LR=0.1 EPOCHS=24" "fewshot rl \"0 1\" 1 $c $n"; done; done
} > "$f"
