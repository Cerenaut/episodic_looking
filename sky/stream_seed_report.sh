#!/bin/bash
# Status of the seeded single-stream round: seed 1 on the M3, seeds 2-5 on RunPod.
# Progress is read from the results files: a finished class has 52 evaluate lines
# (13 evaluation points x 4 test sets), so a seed is complete at 156 across its three classes.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="

echo "--- balance ---"
~/.venvs/sky/bin/python -c "
import tomllib,pathlib,runpod
from runpod.api.graphql import run_graphql_query
runpod.api_key=tomllib.load(open(pathlib.Path.home()/'.runpod/config.toml','rb'))['default']['api_key']
d=run_graphql_query('query { myself { clientBalance currentSpendPerHr } }')['data']['myself']
print('balance \$%.2f   burn \$%.2f/h   pods %d' % (d['clientBalance'], d['currentSpendPerHr'], len(runpod.get_pods())))" 2>/dev/null || echo "balance query failed"

echo "--- clusters ---"
sky status 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^ss-" | \
  awk '{st="?"; for(i=1;i<=NF;i++) if($i=="UP"||$i=="INIT"||$i=="STOPPED") st=$i;
        ad="NONE"; for(i=1;i<=NF;i++) if($i=="(down)") ad=$(i-1);
        printf "  %-8s %-6s autodown %s\n", $1, st, ad}'
[ -z "$(sky status 2>/dev/null | grep -E '^ss-')" ] && echo "  (none)"

echo "--- M3, seed 1 ---"
for c in 3 4 5; do
  s=$(grep "^INFO:root:Step" runs_local/stream_seed/s1_c$c.log 2>/dev/null | awk -F'of ' '$2 ~ /^8000/' | tail -1 | grep -o 'Step: [0-9]*' | grep -o '[0-9]*')
  n=$(grep -ci nan runs_local/stream_seed/s1_c$c.log 2>/dev/null)
  d=$([ -f runs_local/stream_seed/s1_c$c.done ] && echo DONE || echo running)
  printf "  class %s: %8s/768000 steps  %s  nan %s\n" "$c" "${s:-0}" "$d" "${n:-0}"
done

echo "--- RunPod seeds (evaluate lines; 52 per class, 156 per seed) ---"
for seed in 2 3 4 5; do
  c="ss-s$seed"
  sky status "$c" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -qE "^$c .*UP" || { printf "  %-8s not up\n" "$c"; continue; }
  r=$(ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o BatchMode=yes "$c" '
    cd ~/sky_workdir 2>/dev/null || exit 0
    t=0; n=0
    while IFS= read -r -d "" f; do t=$((t+$(grep -c evaluate "$f"))); n=$((n+1)); done < <(find runs_stream_seed -name "results_few-shot.txt" -print0 2>/dev/null)
    fin=0; [ -f runs_local/stream_seed/JOB_COMPLETE ] && fin=1
    echo "$t $n $fin"' 2>/dev/null)
  [ -z "$r" ] && r="? ? 0"
  set -- $r
  printf "  %-8s %4s/156 lines (%s classes)  %s\n" "$c" "${1:-?}" "${2:-?}" "$([ "${3:-0}" = "1" ] && echo COMPLETE || echo running)"
done

echo "--- pulled locally ---"
for seed in 1 2 3 4 5; do
  n=$(find "runs_stream_seed/s$seed" -name 'results_few-shot.txt' 2>/dev/null | wc -l | tr -d ' ')
  ok=0
  while IFS= read -r -d '' f; do [ "$(grep -c evaluate "$f")" = "52" ] && ok=$((ok+1)); done < <(find "runs_stream_seed/s$seed" -name 'results_few-shot.txt' -print0 2>/dev/null)
  printf "  seed %s: %s/3 files, %s complete\n" "$seed" "$n" "$ok"
done
