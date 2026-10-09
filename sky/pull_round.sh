#!/bin/bash
# Pull results off every reachable round pod. Safe to re-run; rsync is incremental.
# Must run before the 120-minute autostop tears a pod down, because nothing is mirrored
# off the pod while it runs.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
mkdir -p runs_seed runs_local/seeds ../cifar_100_pretrain/variants
for seed in 1 2 3 4 5; do for v in pt40 pt12; do
  c="r-$v-s$seed"
  sky status "$c" 2>/dev/null | grep -q UP || { echo "skip $c (not up)"; continue; }
  echo "pulling $c"
  rsync -az --timeout=60 "$c:~/sky_workdir/runs_seed/" runs_seed/ 2>/dev/null || echo "  runs_seed failed"
  rsync -az --timeout=60 "$c:~/sky_workdir/runs_local/seeds/" runs_local/seeds/ 2>/dev/null || echo "  logs failed"
  rsync -az --timeout=60 "$c:~/cifar_100_pretrain/variants/" ../cifar_100_pretrain/variants/ 2>/dev/null || echo "  ckpts failed"
done; done
echo "--- pulled ---"; find runs_seed -name 'results_continual.txt' 2>/dev/null | wc -l | xargs echo "continual results files:"
