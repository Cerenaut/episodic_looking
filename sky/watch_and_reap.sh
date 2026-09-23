#!/bin/bash
# Poll the live round clusters; the moment a job finishes, pull its results and tear the
# pod down. This exists because the first round lost four jobs: they finished, sat idle
# for the two-hour autodown window, and destroyed themselves with their results still on
# the pod. The autodown is the money backstop, not the reaping mechanism; it must never
# be what ends a pod that still holds results.
# Reaping on completion also cuts the bill roughly in half, since no pod bills while idle.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
LOG=runs_local/round
mkdir -p "$LOG" runs_seed runs_local/seeds ../cifar_100_pretrain/variants
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/reap.log"; }

INTERVAL=${INTERVAL:-180}
MAX_PASSES=${MAX_PASSES:-200}
for pass in $(seq 1 "$MAX_PASSES"); do
  live=$(sky status 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^r-" | awk '{print $1}')
  [ -z "$live" ] && { say "no round clusters left; reaper exiting"; exit 0; }
  for c in $live; do
    sky status "$c" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -qE "^$c .*UP" || continue
    fin=$(ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o BatchMode=yes "$c" \
          'grep -c "JOB DONE" ~/sky_workdir/runs_local/seeds/progress.log 2>/dev/null || echo 0' 2>/dev/null)
    [ "${fin:-0}" = "0" ] && continue
    say "$c finished; pulling"
    ok=1
    rsync -az --timeout=120 "$c:~/sky_workdir/runs_seed/" runs_seed/ || ok=0
    rsync -az --timeout=120 "$c:~/sky_workdir/runs_local/seeds/" runs_local/seeds/ || ok=0
    rsync -az --timeout=120 "$c:~/cifar_100_pretrain/variants/" ../cifar_100_pretrain/variants/ || ok=0
    if [ "$ok" = "1" ]; then
      say "$c pulled cleanly; tearing down"
      sky down "$c" -y > /dev/null 2>&1 && say "$c down" || say "$c FAILED to down"
    else
      say "$c PULL FAILED; leaving it up for the next pass (autodown is the backstop)"
    fi
  done
  sleep "$INTERVAL"
done
say "reaper hit MAX_PASSES; clusters may remain"
