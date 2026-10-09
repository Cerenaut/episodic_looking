#!/bin/bash
# Set autodown on every round cluster as it comes UP. The -i/--down flags do not take
# effect when sky launch is given -d, so this must be applied separately; without it a
# lost session would leave ten pods billing indefinitely. Idempotent; safe to re-run.
set -u
cd "$(dirname "$0")/.."
export PATH="$HOME/bin:$PATH"
CLUSTERS=${CLUSTERS:-$(sky status 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^(r-|ss-)" | awk '{print $1}')}
for c in $CLUSTERS; do
  line=$(sky status "$c" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -E "^$c ")
  echo "$line" | grep -q "UP" || { echo "$c: not up yet"; continue; }
  echo "$line" | grep -qE "[0-9]+h \(down\)|[0-9]+m \(down\)" && { echo "$c: autodown already set"; continue; }
  sky autostop "$c" -i 120 --down -y > /dev/null 2>&1 && echo "$c: autodown set" || echo "$c: FAILED to set autodown"
done
