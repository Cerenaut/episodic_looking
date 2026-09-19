#!/bin/bash
# Run every baseline head on the frozen LTM for coarse classes 0 1.
#
#   continual : methods {linear, ncm, flymodel, sdmlp} x orders {3 4 5, 4 5 3, 5 3 4} x seeds {0,1,2}
#   streaming : methods x fine classes {3, 4, 5} x seed 0 (batch size 1)
#
# Usage:  ./run_all_heads.sh <ltm_checkpoint.pth> [extra args passed to cifar_main_head_baselines.py]
#
# Resumable: a run is skipped when its results file already exists and is non-empty. Delete the
# results file (or the run directory) of an interrupted run to redo it.
# Output layout (deterministic, no timestamps, so runs can be skipped and plot_comparison.py can find them):
#   runs/cifar_100/head_<method>_continual_[3, 4, 5]_500/seed_<seed>/results_continual.txt
#   runs/cifar_100/head_<method>_streaming_[3]_500/seed_0/results_streaming.txt
set -u
PY=/Users/gideon/anaconda3/envs/episodic/bin/python

CKPT=${1:?usage: run_all_heads.sh <ltm_checkpoint.pth> [extra args]}
shift
cd "$(dirname "$0")"

METHODS="linear ncm flymodel sdmlp"
COARSE="0 1"

run_if_needed() {  # $1 = run_path, $2 = results suffix, rest = args
    local run_path=$1; shift
    local suffix=$1; shift
    local results="$run_path/results_$suffix.txt"
    if [ -s "$results" ]; then
        echo "skip (exists): $results"
        return 0
    fi
    echo "run: $results"
    "$PY" cifar_main_head_baselines.py --checkpoint "$CKPT" --coarse-classes $COARSE --run-path "$run_path" "$@"
}

for method in $METHODS; do
    for order in "3 4 5" "4 5 3" "5 3 4"; do
        for seed in 0 1 2; do
            name="head_${method}_continual_[${order// /, }]_500"
            run_if_needed "runs/cifar_100/$name/seed_$seed" continual \
                --method "$method" --experiment-type continual --fine-classes $order --seed "$seed" "$@"
        done
    done
done

for method in $METHODS; do
    for fc in 3 4 5; do
        name="head_${method}_streaming_[${fc}]_500"
        run_if_needed "runs/cifar_100/$name/seed_0" streaming \
            --method "$method" --experiment-type streaming --fine-classes "$fc" --seed 0 "$@"
    done
done
