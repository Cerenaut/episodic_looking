#!/bin/bash
# Actions files of the RL STM few-shot final runs (Dave, 14 Oct 2026; plan.md section 3, decision 10): pair A, fine 3,
# 4, 5; seeds 1-10 at N = 1 and 4, seeds 1-5 at N = 16, 64, 400; lr 0.1. Budgets (RL epochs of 4,000 steps x 16
# envs) from the selection rounds (metrics_v2.py --plateau, four-set val curve, seed 1):
#   N=1: 1 (24-epoch round, plateaued; nothing learnt)
#   N=4: 24 (24-epoch round on an L4 still rising to epoch 22; 48-epoch round on an RTX 4090, a different
#        pre-training, plateaued at 6; 24 taken as the larger, within Dave's 24-epoch bound)
#   N=16: 26, N=64: 21 (48-epoch round, budget = first epoch within 0.01 of the best; small gains after 40)
#   N=400: 21 (48-epoch round, plateaued)
# One pod per seed, each with that seed's own STM pre-training shared by its units. Tree runs_v2_fs_rl (fresh, so the
# old runs_v2/pretrain/rl checkpoints are not touched).
#   v2fs-rl-pa-s<seed>  pretrain + 15 units (seeds 1-5) or 6 units (seeds 6-10); longest units first
# Usage (from the repo root): bash sky/rounds/fsfinal/make_actions.sh
set -eu
cd "$(dirname "$0")/../../.."
OUT=sky/rounds/fsfinal
TREE=runs_v2_fs_rl/rl_lr0.1
declare -A EP=([1]=1 [4]=24 [16]=26 [64]=21 [400]=21)
line() {  # env, run_v2.sh args (as one string of shell words)
  local env=$1 args=$2 unit
  unit=$(eval "env $env RUNS=$TREE PRINT_UNIT=1 bash run_v2.sh $args")
  [ -n "$unit" ] || { echo "run_v2.sh refused: $args" >&2; exit 1; }
  echo "$TREE|$env|$args|$unit|"
}
for s in 1 2 3 4 5 6 7 8 9 10; do
  f=$OUT/v2fs-rl-pa-s$s.txt
  [ -e "$f" ] && { echo "exists: $f" >&2; exit 1; }
  if [ $s -le 5 ]; then ns="16 4 64 400 1"; else ns="4 1"; fi
  {
    line "LR=0.1" "pretrain rl \"0 1\" $s"
    for n in $ns; do for c in 3 4 5; do line "LR=0.1 EPOCHS=${EP[$n]}" "fewshot rl \"0 1\" $s $c $n"; done; done
  } > "$f"
done
