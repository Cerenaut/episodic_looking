#!/bin/bash
# Check of the heads' one-shot recall (Dave 16 Nov 2026: SDMLP's Generalisation at n = 1, .45-.83, looks too high;
# leakage?). Reruns seed-0 c 512 stage 2 winners with the new to_current field (other classes' held-out samples
# labelled as the current class), plus the LTM alone; same flags as queue_tune1.sh. Metrics should match tune1/1b.
cd /home/incubator/dev/claude/episodic_looking
export PY=/home/incubator/dev/claude/.venv/bin/python D=runs_ss/mnist_clustered_single_stream/comparisons/leakcheck
mkdir -p $D
export B="--dataset mnist --val-holdout 600 --epochs 1 --max-exposures 2048 --full-first-task --preproc runs_ss/preproc/omniglot_conv_c32_s0/preproc.pt --threads 1 --net transformer --blocks 1 --optimizer adamw --lr 0.0003 --replay-batch 16 --alpha-evict 3 --alpha-replay 1 --n-log-checkpoints 20 --grid-per-epoch 20 --feature-cache runs_ss/mnist_clustered_single_stream/feature_cache --replay-period 256 --replay-steps 64 --buffer priority --replay-capacity 500 --bias-mode what --seed 0 --cluster-size 512"
export H="--model head --head-delay 53500"
run() { name=$1; shift; if $PY -m single_stream.run_tasks $B "$@" --out $D/$name > $D/$name.stdout 2>&1; then
  echo "$(date '+%F %T') done $name" >> $D/queue.log; else echo "$(date '+%F %T') FAILED $name" >> $D/queue.log; fi; }
export -f run
{
for cfg in "lp id,lperm1" "dih id,rot90"; do
  read fam tasks <<< "$cfg"
  echo "run sdmlp_rep64_lr0.5_reset_${fam} --tasks $tasks \$H --head sdmlp --head-repeats 64 --sdmlp-lr 0.5"
  echo "run sdmlp_rep64_lr0.5_keep_${fam} --tasks $tasks \$H --head sdmlp --head-repeats 64 --sdmlp-lr 0.5 --head-no-reset"
  echo "run knn_default_reset_${fam} --tasks $tasks \$H --head knn"
  echo "run fly_lr2_reset_${fam} --tasks $tasks \$H --head fly --fly-lr 2"
  echo "run ltm_${fam} --tasks $tasks --model buffered"
done
} | xargs -P 8 -L 1 bash -c 'eval "$@"' _
echo "$(date '+%F %T') leakcheck done" >> $D/queue.log
