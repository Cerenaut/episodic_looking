#!/bin/bash
# Status of a v2 round (sky/v2_launch.sh): per cluster, the sky state and autostop, and from the pod the number of
# units with job.done out of the units in its actions file, the sentinel, and the last progress line.
#
# Rule (running_experiments_kb.md section 1): "could not read" is never shown as "no progress". A pod that cannot be
# read prints UNREADABLE and the script exits 1; a pod whose job has not started yet prints NOT-STARTED (ssh worked,
# no actions copy on the pod yet). Units done are counted against the total, so a working zero reads "0/5".
# Usage: bash sky/v2_status.sh <round dir>
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
ROUND=${1:?round dir}
[ -d "$ROUND/actions" ] || { echo "UNREADABLE: no $ROUND/actions (not a round dir)"; exit 1; }
strip() { perl -pe 's/\e\[[0-9;]*[mK]//g'; }
rc=0
echo "=== $ROUND  $(date '+%Y-%m-%d %H:%M:%S') ==="
printf '%-18s %-9s %-11s %-9s %-13s %s\n' cluster sky autostop units sentinel "last progress line"
for af in "$ROUND"/actions/*.txt; do
  [ -f "$af" ] || { echo "UNREADABLE: no actions files in $ROUND/actions"; rc=1; continue; }
  c=$(basename "$af" .txt)
  total=$(grep -c . "$af")
  if [ -e "$ROUND/reaped/$c" ]; then
    printf '%-18s %-9s %-11s %-9s %-13s %s\n' "$c" REAPED - "$total/$total" verified "$(cat "$ROUND/reaped/$c")"; continue
  fi
  line=$(sky status "$c" 2>/dev/null | strip | awk -v c="$c" '$1==c')
  if [ -z "$line" ]; then printf '%-18s %-9s %s\n' "$c" ABSENT "not in sky status and not reaped: CHECK"; rc=1; continue; fi
  st=$(echo "$line" | grep -owE 'UP|INIT|STOPPED' | head -1); st=${st:-UNPARSED}
  au=$(echo "$line" | grep -oE '[0-9]+[hm] \(down\)' | head -1); au=${au:-NONE}
  if [ "$st" != UP ]; then printf '%-18s %-9s %-11s\n' "$c" "$st" "$au"; continue; fi
  # One ssh call (60 s cap via perl alarm: macOS has no timeout(1)); the remote prints exactly one line "<done> <total> <sentinel> <last line>" or NOT-STARTED.
  r=$(perl -e 'alarm shift; exec @ARGV' 60 ssh -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no "$c" "bash -s" 2>/dev/null <<EOF
cd ~/sky_workdir 2>/dev/null || { echo NOWORKDIR; exit 0; }
P=$ROUND/pods/$c
[ -f "\$P/actions.txt" ] || { echo NOT-STARTED; exit 0; }
d=0; t=0
while IFS='|' read -r tree env args unit flag; do [ -z "\$tree" ] && continue; t=\$((t+1)); [ -f "\$unit/job.done" ] && d=\$((d+1)); done < "\$P/actions.txt"
s=running; [ -f "\$P/JOB_FAILED" ] && s=JOB_FAILED; [ -f "\$P/JOB_COMPLETE" ] && s=JOB_COMPLETE
echo "\$d \$t \$s \$(tail -1 "\$P/progress.log" 2>/dev/null | cut -c1-110)"
EOF
)
  case "$r" in
    NOT-STARTED|NOWORKDIR) printf '%-18s %-9s %-11s %-9s %-13s %s\n' "$c" UP "$au" "?/$total" "$r" "(setup still running?)" ;;
    *)
      set -f; set -- $r; set +f   # no globbing: the progress line holds [brackets]
      if [[ "${1:-}" =~ ^[0-9]+$ ]] && [[ "${2:-}" =~ ^[0-9]+$ ]] && [ -n "${3:-}" ]; then
        printf '%-18s %-9s %-11s %-9s %-13s %s\n' "$c" UP "$au" "$1/$2" "$3" "${r#* * * }"
      else
        printf '%-18s %-9s %-11s %s\n' "$c" UP "$au" "UNREADABLE (ssh failed or unparsable reply: '${r:0:60}')"; rc=1
      fi ;;
  esac
done
exit $rc
