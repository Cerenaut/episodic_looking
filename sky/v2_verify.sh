#!/bin/bash
# The destructive predicate of sky/v2_reap.sh, kept separate so it can be tested on fake trees
# (sky/test_v2_verify.sh). Exit 0 ONLY when the pulled results of one pod job are verifiably on this machine;
# anything missing, unreadable or unexpected is exit 1. It never touches a pod.
#
# Checks, on local files only. Unit paths are relative to ROOT (the pull root, default the repo); ROUND is relative
# to ROOT or absolute (the reaper passes the repo's round dir, which holds actions/ and pods/, as an absolute path, so
# that the unit trees can be pulled to another root: sky/v2_reap.sh PULL_ROOT).
#   1. $ROUND/pods/$JOB/JOB_COMPLETE was pulled (the pod finished cleanly), and no JOB_FAILED next to it.
#   2. The pulled actions.txt is byte-identical to the launcher's copy $ROUND/actions/$JOB.txt.
#   3. manifest.tsv (size<TAB>sha256<TAB>path, written on the pod after the last unit) is non-empty, every line
#      parses, every path lies under a unit dir of the actions file, and every file exists here with the same size
#      and sha256.
#   4. Every unit of the actions file: its local file set equals the manifest's entries for it (no extra or stale
#      file here, none missing); job.done; at least one results_*.txt (not _val) and, unless its environment sets
#      VAL_HOLDOUT=0, at least one results_*_val.txt, each with an "evaluate" line; and for STM units (rl, actor) the
#      checkpoint: pretrain always writes stm_pretrain.pth; continual, stream and few-shot write a .pth unless the
#      environment sets SAVE_STM=0.
# Paths with a component starting with '.' are ignored on both sides (the pod job leaves them out of the manifest):
# run_v2.sh's .lock directory, and a .DS_Store that Finder may drop on the Mac. They hold no results.
#
# --partial (for a JOB_FAILED pod, whose files the reaper pulls once): exit 0 only when every file the pod's manifest
# lists is here with the same size and sha256, i.e. nothing the pod produced would be lost by tearing it down. Checks
# 2 and 3 only (the pulled actions.txt = the launcher's copy; manifest.tsv present, every line parses, every path
# under a unit of the actions file, every file here with the same size and sha256). No completeness checks: no
# JOB_COMPLETE, job.done, results or checkpoints required, and JOB_FAILED is expected. An EMPTY manifest passes (the
# pod job writes one when no unit dir exists, e.g. a job that failed in its pre-flight); a MISSING one fails.
# Usage: bash sky/v2_verify.sh [--partial] <round dir> <job> [root]
set -u
PARTIAL=0
[ "${1:-}" = --partial ] && { PARTIAL=1; shift; }
ROUND=${1:?round dir}
JOB=${2:?job (cluster) name}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${3:-$(cd "$HERE/.." && pwd)}
# shellcheck source=sky/v2_lib.sh
. "$HERE/v2_lib.sh" || { echo "VERIFY FAIL: cannot read $HERE/v2_lib.sh"; exit 1; }
cd "$ROOT" || { echo "VERIFY FAIL: cannot cd to $ROOT"; exit 1; }
P=$ROUND/pods/$JOB
AC=$ROUND/actions/$JOB.txt
bad=0
no() { echo "VERIFY FAIL [$JOB]: $*"; bad=1; }
TAB=$(printf '\t')

if [ "$PARTIAL" = 0 ]; then
  [ -f "$P/JOB_COMPLETE" ] || no "no JOB_COMPLETE in $P"
  [ -e "$P/JOB_FAILED" ] && no "JOB_FAILED present in $P"
fi
[ -s "$AC" ] || no "launcher's actions copy $AC missing or empty"
if [ "$PARTIAL" = 1 ] && [ ! -e "$P/actions.txt" ] && [ -f "$P/manifest.tsv" ] && [ ! -s "$P/manifest.tsv" ]; then
  # a job that failed before copying its actions file (e.g. no UNIT_TIMEOUT): no unit ran, the manifest is empty
  echo "VERIFY RESULT [$JOB]: PARTIAL OK (the job failed before copying its actions file; empty manifest: no files on the pod)"
  exit 0
fi
[ -s "$P/actions.txt" ] || no "pulled actions.txt missing or empty"
if [ "$bad" = 0 ]; then cmp -s "$AC" "$P/actions.txt" || no "pulled actions.txt differs from the launcher's copy"; fi
if [ "$PARTIAL" = 1 ]; then [ -f "$P/manifest.tsv" ] || no "manifest.tsv missing"
else [ -s "$P/manifest.tsv" ] || no "manifest.tsv missing or empty"; fi
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
  size=${mline%%"$TAB"*}; rest=${mline#*"$TAB"}
  hash=${rest%%"$TAB"*}; path=${rest#*"$TAB"}
  if [ "$size" = "$mline" ] || [ "$hash" = "$rest" ] || ! [[ "$size" =~ ^[0-9]+$ ]] || ! [[ "$hash" =~ ^[0-9a-f]{64}$ ]] \
     || [ -z "$path" ]; then no "unparsable manifest line: $mline"; continue; fi
  under=0
  for u in "${units[@]}"; do case "$path" in "$u"/*) under=1; break ;; esac; done
  [ "$under" = 1 ] || { no "manifest path not under any unit: $path"; continue; }
  [ -f "$path" ] || { no "missing locally: $path"; continue; }
  lsize=$(wc -c < "$path" 2>/dev/null | tr -d ' ')
  [ "$lsize" = "$size" ] || { no "size differs ($lsize here, $size on pod): $path"; continue; }
  lhash=$(sha256 "$path")
  [ "$lhash" = "$hash" ] || { no "sha256 differs (${lhash:-unreadable} here, $hash on pod): $path"; continue; }
  nfiles=$((nfiles + 1))
done < "$P/manifest.tsv"

if [ "$PARTIAL" = 1 ]; then
  if [ "$bad" = 0 ]; then
    echo "VERIFY RESULT [$JOB]: PARTIAL OK ($nfiles files of the pod's manifest are here: size, sha256; completeness not checked)"
    exit 0
  fi
  echo "VERIFY RESULT [$JOB]: FAIL"; exit 1
fi

# 4. units
while IFS='|' read -r tree env args unit flag || [ -n "$tree" ]; do
  [ -z "$tree" ] && continue
  eval "set -- $args" 2>/dev/null || { no "unparsable args: $args"; continue; }
  setting=${1:-}; model=${2:-}
  env_value "$env" X > /dev/null || { no "unparsable environment field: $env"; continue; }
  [ -d "$unit" ] || { no "unit dir missing here: $unit"; continue; }
  # local file set == manifest entries for this unit (sorted, newline-separated; a path with a newline would show up
  # as a mismatch, i.e. fail, never pass)
  lset=$(find "$unit" -name '.*' -prune -o -type f -print 2>/dev/null | LC_ALL=C sort)
  mset=$(awk -F'\t' -v u="$unit/" 'index($3, u) == 1 { print $3 }' "$P/manifest.tsv" | LC_ALL=C sort)
  [ -n "$mset" ] || no "no manifest entries for $unit"
  if [ "$lset" != "$mset" ]; then
    extra=$(LC_ALL=C comm -23 <(printf '%s\n' "$lset") <(printf '%s\n' "$mset") | grep . | head -5)
    gone=$(LC_ALL=C comm -13 <(printf '%s\n' "$lset") <(printf '%s\n' "$mset") | grep . | head -5)
    [ -n "$extra" ] && no "files here that are not in the pod's manifest (extra or stale): $(echo "$extra" | tr '\n' ';')"
    [ -n "$gone" ] && no "manifest files not here: $(echo "$gone" | tr '\n' ';')"
    [ -z "$extra$gone" ] && no "local file set differs from the manifest: $unit"
  fi
  [ -f "$unit/job.done" ] || { no "no job.done: $unit"; continue; }
  main=0; val=0
  while IFS= read -r -d '' f; do
    grep -q evaluate "$f" 2>/dev/null || continue
    case "$f" in *_val.txt) val=$((val + 1)) ;; *) main=$((main + 1)) ;; esac
  done < <(find "$unit" -type f -name 'results_*.txt' ! -name '*_images.txt' -print0 2>/dev/null)
  [ "$main" -ge 1 ] || no "no results_*.txt with an evaluate line: $unit"
  [ "$(env_value "$env" VAL_HOLDOUT)" = 0 ] || [ "$val" -ge 1 ] || no "no results_*_val.txt with an evaluate line: $unit"
  case "$model/$setting" in
    rl/pretrain|actor/pretrain) [ -s "$unit/stm_pretrain.pth" ] || no "no stm_pretrain.pth: $unit" ;;
    rl/continual|actor/continual|rl/stream|actor/stream|rl/fewshot|actor/fewshot)
      if [ "$(env_value "$env" SAVE_STM)" != 0 ]; then
        [ -n "$(find "$unit" -maxdepth 1 -name '*.pth' -size +0 -print -quit)" ] || no "no STM checkpoint (.pth): $unit"
      fi ;;
  esac
done < "$P/actions.txt"

if [ "$bad" = 0 ] && [ "$nfiles" -gt 0 ]; then
  echo "VERIFY RESULT [$JOB]: OK (${#units[@]} units, $nfiles files match the pod's manifest: size, sha256, file set)"
  exit 0
fi
echo "VERIFY RESULT [$JOB]: FAIL"
exit 1
