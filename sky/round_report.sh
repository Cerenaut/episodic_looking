#!/bin/bash
# One-shot status of everything in flight: the RunPod seed round, the M3 single-stream
# run, and the account balance. Written for the hourly update.
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
print('balance \$%.2f   burn \$%.2f/h' % (d['clientBalance'], d['currentSpendPerHr']))" 2>/dev/null || echo "balance query failed"

echo "--- clusters ---"
sky status 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^r-" | \
  awk '{st="?"; for(i=1;i<=NF;i++) if($i=="UP"||$i=="INIT"||$i=="STOPPED") st=$i;
        ad="none"; for(i=1;i<=NF;i++) if($i=="(down)") ad=$(i-1)" (down)";
        printf "%-12s %-6s autodown %s\n", $1, st, ad}'

echo "--- job progress (evaluate lines; pretrain 48=pt12 160=pt40, each order 144, 432 total) ---"
for seed in 1 2 3 4 5; do for v in pt40 pt12; do
  c="r-$v-s$seed"
  sky status "$c" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -qE "^$c .*UP" || { printf "%-12s %s\n" "$c" "not up"; continue; }
  r=$(ssh -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o BatchMode=yes "$c" '
    cd ~/sky_workdir 2>/dev/null || exit 0
    p=0; for f in $(find runs_seed -name "results_pretrain.txt" 2>/dev/null); do p=$(grep -c evaluate "$f"); done
    t=0; n=0; for f in $(find runs_seed -name "results_continual.txt" 2>/dev/null); do t=$((t+$(grep -c evaluate "$f"))); n=$((n+1)); done
    d=$(ls runs_local/seeds/*.done 2>/dev/null | wc -l)
    echo "$p $t $n $d"' 2>/dev/null)
  [ -z "$r" ] && r="? ? ? ?"
  set -- $r
  printf "%-12s pretrain %-6s continual %-4s/432 (%s orders)  stages %s/4\n" "$c" "${1:-?}" "${2:-?}" "${3:-?}" "${4:-?}"
done; done

echo "--- M3 single-stream (lands separately) ---"
for c in 3 4 5; do
  s=$(grep "^INFO:root:Step" runs_local/stream_full/c$c.log 2>/dev/null | tail -1 | grep -o '[0-9]*' | head -1)
  n=$(grep -ci nan runs_local/stream_full/c$c.log 2>/dev/null)
  printf "  class %s: %s/768000 training steps, nan lines %s\n" "$c" "${s:-?}" "${n:-?}"
done
