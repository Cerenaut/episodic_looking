#!/bin/bash
# Compact status of the ten-job round: cluster state, per-job progress, and an ETA.
# Progress is read from the results files rather than from logs: a finished continual
# order has 144 evaluate lines (36 epochs x 4 test sets), a pt12 pre-training 48 and a
# pt40 one 160, so the fraction done is exact rather than inferred.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
OUT=${1:-/dev/stdout}
probe() {  # $1 = cluster
  local c=$1 v seed pt
  case "$c" in *pt40*) pt=160 ;; *) pt=48 ;; esac
  timeout 60 ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$c" "
    cd ~/sky_workdir 2>/dev/null || { echo 'NOWORKDIR'; exit 0; }
    pre=\$(find runs_seed -name 'results_pretrain.txt' 2>/dev/null | head -1)
    npre=\$([ -n \"\$pre\" ] && grep -c evaluate \"\$pre\" 2>/dev/null || echo 0)
    tot=0
    for f in \$(find runs_seed -name 'results_continual.txt' 2>/dev/null); do
      n=\$(grep -c evaluate \"\$f\" 2>/dev/null || echo 0); tot=\$((tot+n))
    done
    nord=\$(find runs_seed -name 'results_continual.txt' 2>/dev/null | wc -l)
    done_markers=\$(ls runs_local/seeds/*.done 2>/dev/null | wc -l)
    echo \"\$npre \$tot \$nord \$done_markers\"
  " 2>/dev/null || echo "UNREACHABLE"
}
{
echo "=== round status $(date '+%Y-%m-%d %H:%M:%S') ==="
printf "%-14s %-10s %-14s %-16s %-8s\n" "cluster" "state" "pretrain" "continual" "stages"
for seed in 1 2 3 4 5; do for v in pt40 pt12; do
  c="r-$v-s$seed"
  st=$(sky status "$c" 2>/dev/null | awk -v c="$c" '$1==c {print $6}' | head -1)
  [ -z "$st" ] && st="-"
  case "$v" in pt40) pt=160 ;; *) pt=48 ;; esac
  if [ "$st" = "UP" ]; then r=$(probe "$c"); else r=""; fi
  if [ -n "$r" ] && [ "$r" != "UNREACHABLE" ] && [ "$r" != "NOWORKDIR" ]; then
    set -- $r; npre=${1:-0}; tot=${2:-0}; nord=${3:-0}; dn=${4:-0}
    printf "%-14s %-10s %-14s %-16s %-8s\n" "$c" "$st" "$npre/$pt" "$tot/432 ($nord ord)" "$dn/4"
  else
    printf "%-14s %-10s %-14s %-16s %-8s\n" "$c" "$st" "${r:--}" "-" "-"
  fi
done; done
} > "$OUT" 2>&1
