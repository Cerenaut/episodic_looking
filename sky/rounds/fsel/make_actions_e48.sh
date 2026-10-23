#!/bin/bash
# Actions files of the RL STM few-shot budget selection at 48 epochs (Dave, 14 Oct 2026, option 2): the 24-epoch
# selection (make_actions.sh, tree runs_v2_pilot_fewshot/rl_lr0.1) had not plateaued at N = 4/16/64/400 (plateau rule:
# extend to 48), so those N are re-run from scratch at 48 epochs; N = 1 plateaued (budget 1) and is not re-run.
# Pair A, seed 1, fine 3, 4, 5, lr 0.1, validation every epoch; budget per N from metrics_v2.py --plateau.
# Four pods, one per N, each with its own STM pre-training (seed 1) shared by its 3 units (one batch at NPROC=3).
# One tree per N, runs_v2_pilot_fewshot_e48_n<N>/rl_lr0.1, so the four pods' seed-1 pre-training directories do not
# collide on the pull (the parent differs from the 24-epoch tree; the selection tool reads lr from the tree name).
#   v2fsel48-rl-pa-n<N>  pretrain + 3 few-shot units
# Usage (from the repo root): bash sky/rounds/fsel/make_actions_e48.sh
set -eu
cd "$(dirname "$0")/../../.."
OUT=sky/rounds/fsel
line() {  # env, run_v2.sh args (as one string of shell words)
  local env=$1 args=$2 unit
  unit=$(eval "env $env RUNS=$TREE PRINT_UNIT=1 bash run_v2.sh $args")
  [ -n "$unit" ] || { echo "run_v2.sh refused: $args" >&2; exit 1; }
  echo "$TREE|$env|$args|$unit|"
}
for n in 400 64 16 4; do
  TREE=runs_v2_pilot_fewshot_e48_n$n/rl_lr0.1
  f=$OUT/v2fsel48-rl-pa-n$n.txt
  [ -e "$f" ] && { echo "exists: $f" >&2; exit 1; }
  {
    line "LR=0.1" "pretrain rl \"0 1\" 1"
    for c in 3 4 5; do line "LR=0.1 EPOCHS=48" "fewshot rl \"0 1\" 1 $c $n"; done
  } > "$f"
done
