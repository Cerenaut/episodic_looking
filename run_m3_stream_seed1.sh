#!/bin/bash
# Seed 1 of the seeded single-stream round, on the M3. Seeds 2-5 run on RunPod.
set -u
cd "$(dirname "$0")"
export PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec ./sky/run_stream_seed.sh 1
