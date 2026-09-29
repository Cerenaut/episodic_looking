#!/bin/bash
# NOT IN USE since 14:05 2026-09-20: Gideon freed Walter (M3 only from now on); run_m3_pt40.sh ran the orders instead.
# Kept as the template for splitting a job's orders across two GPUs. Bug fixed after the fact: the Walter-free check used
# `pgrep -c ... || echo unreachable`, and pgrep exits 1 on zero matches, so the check never passed.
# Item 19, rebalanced 2026-09-20 10:10. Profiling showed the STM runs are GPU-bound (two ResNet-18 forwards x 16 images per
# step), so N processes on one GPU take N times longer each: parallelism buys nothing, and Walter's GTX 1060 is ~2x the M3's GPU.
# Plan: the M3 pre-trains the STM for 40 epochs alone (~3.5 h), then orders 3,4,5 and 4,5,3 run on Walter (2 in parallel,
# ~4.5 h) and order 5,3,4 on the M3 (~5 h); Walter's results are mirrored back into runs_p0_pt40/ here. Everything lands
# Sunday evening instead of Monday 06:30. Walter orders launch only if they can finish before the 21:00 curfew and Walter is
# free (A.1 repeat 2 done); otherwise all three orders run here as run_m3_pt40.sh did.
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/pt40
mkdir -p "$LOG"
E13=$CK/cifar_100_subclasses_12_e13.pth
STM=$CK/variants/stm_rl_fixed_e13_pt40.pth
RUN="$PY cifar_main_stm_training.py --coarse-classes 0 1 --eval-bias mean --evaluate-steps 800 --ltm-checkpoint $E13 --stm-checkpoint $STM --run-root runs_p0_pt40"
SSH="ssh -o ClearAllForwardings=yes -o ConnectTimeout=20"
W=g_walter
WALTER_ORDER_MIN=$((36 * 8 + 10))   # 2 orders in parallel on Walter: ~7.3 min/epoch each, plus margin
LATEST_LAUNCH=$((16 * 60 + 30))     # must start by 16:30 to finish before 21:00
step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
now_min() { echo $((10#$(date +%H) * 60 + 10#$(date +%M))); }

step "M3 free; starting item 19 (split plan)"
if [ ! -f "$LOG/pretrain.done" ]; then
  step "pt40: STM pre-training 40 epochs on e13 (M3, alone)"
  $RUN --experiment-type pretrain --fine-classes 1 2 --epochs 40 > "$LOG/pretrain.log" 2>&1 && touch "$LOG/pretrain.done" || { step "FAILED: pre-training"; exit 1; }
fi

# Wait for Walter to be free (A.1 repeat 2 orders), but not past the latest launch time.
walter_ok=0
while [ $(now_min) -le $LATEST_LAUNCH ]; do
  busy=$($SSH $W 'pgrep -f cifar_main_stm_training | wc -l | tr -d " "' 2>/dev/null) || busy=unreachable
  if [ "$busy" = "0" ]; then walter_ok=1; break; fi
  step "Walter busy ($busy STM processes) or unreachable; waiting"
  sleep 300
done
if [ $walter_ok -eq 1 ] && [ $(( $(now_min) + WALTER_ORDER_MIN )) -gt $((21 * 60)) ]; then walter_ok=0; fi

if [ $walter_ok -eq 1 ]; then
  step "launching orders 3,4,5 and 4,5,3 on Walter"
  rsync -az -e "$SSH" "$STM" $W:~/Dev/cifar_100_pretrain/variants/ \
    && rsync -az -e "$SSH" run_walter_pt40_orders.sh $W:~/Dev/episodic_looking/ \
    && $SSH $W 'cd ~/Dev/episodic_looking && mkdir -p runs_local/pt40 && chmod +x run_walter_pt40_orders.sh && nohup ./run_walter_pt40_orders.sh > runs_local/pt40/launcher.log 2>&1 < /dev/null &' \
    || { step "Walter launch FAILED; falling back to all orders on the M3"; walter_ok=0; }
fi

if [ $walter_ok -eq 1 ]; then
  m3_orders="5 3 4"
else
  step "curfew or Walter unavailable: all three orders on the M3"
  m3_orders="3 4 5|4 5 3|5 3 4"
fi
IFS='|' read -ra M3_ORDERS <<< "$m3_orders"
for order in "${M3_ORDERS[@]}"; do
  tag=$(echo "$order" | tr -d ' ')
  [ -f "$LOG/continual_$tag.done" ] && continue
  step "pt40: continual order $order (M3)"
  ( $RUN --experiment-type continual --fine-classes $order > "$LOG/continual_$tag.log" 2>&1 && touch "$LOG/continual_$tag.done" && step "order $order finished" || step "FAILED: order $order" ) &
done
wait

if [ $walter_ok -eq 1 ]; then
  step "M3 order done; waiting for Walter orders"
  while :; do
    n=$($SSH $W 'ls ~/Dev/episodic_looking/runs_local/pt40/continual_*.done 2>/dev/null | wc -l' 2>/dev/null || echo 0)
    [ "$n" -ge 2 ] && break
    sleep 300
  done
  step "mirroring Walter results into runs_p0_pt40/"
  rsync -az -e "$SSH" $W:~/Dev/episodic_looking/runs_p0_pt40/ runs_p0_pt40/ \
    && rsync -az -e "$SSH" --include='continual_*' --exclude='*' $W:~/Dev/episodic_looking/runs_local/pt40/ "$LOG/walter/" \
    && step "Walter results mirrored" || step "FAILED: mirror from Walter (rerun the rsync by hand)"
fi
step "ALL DONE"
