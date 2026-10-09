#!/bin/bash
# Launch the ten-job cloud seed round, one pod per (variant, seed), detached.
# pt40 jobs go first because they are the long pole (~4.4 h against ~2.8 h).
# Each pod auto-tears-down after 120 idle minutes, so a lost session cannot leave one
# running; that window is also the deadline for pulling results, since nothing is
# mirrored off the pod. sky/pull_round.sh does the pulling.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
LOG=runs_local/round
mkdir -p "$LOG"
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/launch.log"; }

for seed in 1 2 3 4 5; do
  for v in ref_pt40 ref_pt12; do
    c="r-${v#ref_}-s$seed"
    if sky status "$c" 2>/dev/null | grep -qE "UP|INIT"; then step "$c already up, skipping"; continue; fi
    step "launching $c ($v seed $seed)"
    nohup sky launch -c "$c" sky/stm_round.yaml \
        --env VARIANT="$v" --env SEED="$seed" --env NPROC=3 \
        -i 120 --down -y -d > "$LOG/$c.launch.log" 2>&1
    sleep 5
  done
done
step "all launch commands issued"
