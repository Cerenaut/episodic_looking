#!/bin/bash
# Run a grid of single_stream.run_stream jobs, P at a time.
# Usage: bash single_stream/scripts/run_grid.sh <out_dir> <jobs_file> [P]
# jobs_file: one job per line, "<name> <run_stream args...>"; outputs go to <out_dir>/<name>. Lines starting with # skipped.
OUT=$1; JOBS=$2; P=${3:-4}
PY=/home/incubator/dev/claude/.venv/bin/python
mkdir -p "$OUT"
cp "$JOBS" "$OUT/jobs.txt"
grep -v '^#' "$JOBS" | grep -v '^ *$' | while read -r name args; do echo "$name $args"; done | \
  xargs -P "$P" -L 1 bash -c 'name=$0; if '"$PY"' -m single_stream.run_stream "$@" --out '"$OUT"'/$name > '"$OUT"'/$name.stdout 2>&1; then
      echo "$(date "+%F %T") done $name $(grep -o "final.*" '"$OUT"'/$name/log.txt | tail -1)"; else echo "$(date "+%F %T") FAILED $name"; fi' \
  >> "$OUT/queue.log"
echo "$(date "+%F %T") all done" >> "$OUT/queue.log"
