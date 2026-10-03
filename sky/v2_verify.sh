#!/bin/bash
# The destructive predicate of sky/v2_reap.sh, kept separate so it can be tested on fake trees
# (sky/test_v2_verify.sh). Exit 0 ONLY when the pulled results of one pod job are verifiably on this machine;
# anything missing, unreadable or unexpected is exit 1. It never touches a pod.
#
# Checks, on local files only (all paths relative to ROOT, default the repo):
#   1. $ROUND/pods/$JOB/JOB_COMPLETE was pulled (the pod finished cleanly), and no JOB_FAILED next to it.
#   2. The pulled actions.txt is byte-identical to the launcher's copy $ROUND/actions/$JOB.txt.
#   3. manifest.tsv (size<TAB>path, written on the pod after the last unit) is non-empty, every line parses, every
#      path lies under a unit dir of the actions file, and every file exists here with the same size.
#   4. Every unit of the actions file: job.done; at least one results_*.txt (not _val) and, unless VAL_HOLDOUT=0 is
#      in its environment, at least one results_*_val.txt, each with an "evaluate" line; at least one manifest
#      entry; and for STM units (rl, actor) that train, the checkpoint (pretrain: stm_pretrain.pth; others: a .pth).
# Usage: bash sky/v2_verify.sh <round dir> <job> [root]
set -u
ROUND=${1:?round dir}
JOB=${2:?job (cluster) name}
ROOT=${3:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT" || { echo "VERIFY FAIL: cannot cd to $ROOT"; exit 1; }
P=$ROUND/pods/$JOB
bad=0
no() { echo "VERIFY FAIL [$JOB]: $*"; bad=1; }

[ -f "$P/JOB_COMPLETE" ] || no "no JOB_COMPLETE in $P"
[ -e "$P/JOB_FAILED" ] && no "JOB_FAILED present in $P"
[ -s "$ROUND/actions/$JOB.txt" ] || no "launcher's actions copy $ROUND/actions/$JOB.txt missing or empty"
[ -s "$P/actions.txt" ] || no "pulled actions.txt missing or empty"
if [ "$bad" = 0 ]; then cmp -s "$ROUND/actions/$JOB.txt" "$P/actions.txt" || no "pulled actions.txt differs from the launcher's copy"; fi
[ -s "$P/manifest.tsv" ] || no "manifest.tsv missing or empty"
[ "$bad" = 0 ] || { echo "VERIFY RESULT [$JOB]: FAIL"; exit 1; }

# Unit dirs of the actions file
units=()
while IFS='|' read -r tree env args unit flag || [ -n "$tree" ]; do
  [ -z "$tree" ] && continue
  [ -n "$unit" ] || { no "empty unit dir in actions"; continue; }
  units+=("$unit")
done < "$P/actions.txt"
[ "${#units[@]}" -gt 0 ] || { no "no units in actions.txt"; echo "VERIFY RESULT [$JOB]: FAIL"; exit 1; }

# 3. manifest
nfiles=0
while IFS= read -r mline || [ -n "$mline" ]; do
  [ -z "$mline" ] && continue
  size=${mline%%$'\t'*}
  path=${mline#*$'\t'}
  if [ "$size" = "$mline" ] || ! [[ "$size" =~ ^[0-9]+$ ]] || [ -z "$path" ]; then no "unparsable manifest line: $mline"; continue; fi
  under=0
  for u in "${units[@]}"; do case "$path" in "$u"/*) under=1; break ;; esac; done
  [ "$under" = 1 ] || { no "manifest path not under any unit: $path"; continue; }
  [ -f "$path" ] || { no "missing locally: $path"; continue; }
  lsize=$(wc -c < "$path" 2>/dev/null | tr -d ' ')
  [ "$lsize" = "$size" ] || { no "size differs ($lsize here, $size on pod): $path"; continue; }
  nfiles=$((nfiles + 1))
done < "$P/manifest.tsv"

# 4. units
while IFS='|' read -r tree env args unit flag || [ -n "$tree" ]; do
  [ -z "$tree" ] && continue
  eval "set -- $args" 2>/dev/null || { no "unparsable args: $args"; continue; }
  setting=${1:-}; model=${2:-}
  [ -f "$unit/job.done" ] || { no "no job.done: $unit"; continue; }
  grep -qF "$(printf '\t')$unit/" "$P/manifest.tsv" || no "no manifest entries for $unit"
  main=0; val=0
  while IFS= read -r -d '' f; do
    grep -q evaluate "$f" 2>/dev/null || continue
    case "$f" in *_val.txt) val=$((val + 1)) ;; *) main=$((main + 1)) ;; esac
  done < <(find "$unit" -type f -name 'results_*.txt' ! -name '*_images.txt' -print0 2>/dev/null)
  [ "$main" -ge 1 ] || no "no results_*.txt with an evaluate line: $unit"
  case " $env " in *" VAL_HOLDOUT=0 "*) ;; *) [ "$val" -ge 1 ] || no "no results_*_val.txt with an evaluate line: $unit" ;; esac
  case "$model/$setting" in
    rl/pretrain|actor/pretrain) [ -s "$unit/stm_pretrain.pth" ] || no "no stm_pretrain.pth: $unit" ;;
    rl/continual|actor/continual|rl/stream|actor/stream|rl/fewshot|actor/fewshot)
      [ -n "$(find "$unit" -maxdepth 1 -name '*.pth' -size +0 -print -quit)" ] || no "no STM checkpoint (.pth): $unit" ;;
  esac
done < "$P/actions.txt"

if [ "$bad" = 0 ] && [ "$nfiles" -gt 0 ]; then
  echo "VERIFY RESULT [$JOB]: OK (${#units[@]} units, $nfiles files match the pod's manifest)"
  exit 0
fi
echo "VERIFY RESULT [$JOB]: FAIL"
exit 1
