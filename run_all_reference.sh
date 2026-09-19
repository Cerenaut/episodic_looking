#!/bin/bash
# Full local reproduction + comparison pipeline (Mac M3, MPS).
# Resumable: each step is skipped if its output already exists.
# Order is chosen so cheap, informative results land first; the slow STM RL runs go last.
#
#   1. LTM pretraining (40 epochs, ~1 h)          -> ../cifar_100_pretrain/cifar_100_subclasses_12_e{N}.pth
#   2. checkpoint selection (best test acc on fine 1,2) -> cifar_100_subclasses_12_e11_31.1.pth (name expected by scripts)
#   3. LTM-only continual fine-tuning, 3 orders (~40 min)
#   4. frozen-encoding head baselines sweep (~10 min)
#   5. plot / summary (partial)
#   6. STM pretraining, 12 epochs (~1 h; paper checkpoint name says e12, repo .sh says 40)
#   7. STM continual, 3 orders, sequential (~9 h at ~5 min/epoch)
#   8. plot / summary (final)
set -u
cd "$(dirname "$0")"
PY=/Users/gideon/anaconda3/envs/episodic/bin/python
export PYTORCH_ENABLE_MPS_FALLBACK=1
CK=../cifar_100_pretrain
LOG=runs_local/reference
mkdir -p "$LOG"
CC="0 1"                                   # coarse classes: aquatic mammals (0) and fish (1); paper's default pair
LTM_CKPT=$CK/cifar_100_subclasses_12_e11_31.1.pth
STM_CKPT=$CK/cifar_100_cc56_subclasses_12_stm_pretrain_e12.pth

step() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG/progress.log"; }
fail() { step "FAILED: $*"; exit 1; }

# 1. LTM pretraining
if [ ! -f "$CK/cifar_100_subclasses_12_e40.pth" ]; then
  step "1/8 LTM pretraining, 40 epochs"
  $PY pretrain_ltm.py --epochs 40 > "$LOG/ltm_pretrain.log" 2>&1 || fail "LTM pretraining"
fi

# 2. checkpoint selection
if [ ! -f "$LTM_CKPT" ]; then
  step "2/8 selecting LTM checkpoint (best acc_12)"
  BEST=$($PY - <<'EOF'
import pandas as pd
df = pd.read_csv("../cifar_100_pretrain/cifar_100_subclasses_12_epochs.csv")
r = df.loc[df["acc_12"].idxmax()]
print(int(r["epoch"]))
EOF
) || fail "checkpoint selection"
  cp "$CK/cifar_100_subclasses_12_e${BEST}.pth" "$LTM_CKPT" || fail "copy checkpoint"
  { echo "selected_epoch=$BEST"; grep -E "^epoch|^${BEST}," "$CK/cifar_100_subclasses_12_epochs.csv"; } | tee "$LOG/selected_epoch.txt"
fi

# 3. LTM-only continual, 3 orders (sequential)
for order in "3 4 5" "4 5 3" "5 3 4"; do
  tag=$(echo "$order" | tr -d ' ')
  if [ ! -f "$LOG/ltm_continual_$tag.done" ]; then
    step "3/8 LTM-only continual, order $order (lr 0.001)"
    $PY cifar_main_ltm_fine_tuning.py --experiment-type continual --fine-classes $order --learning-rate 0.001 --coarse-classes $CC \
      > "$LOG/ltm_continual_$tag.log" 2>&1 && touch "$LOG/ltm_continual_$tag.done" || fail "LTM continual $order"
  fi
done

# 4. head baselines sweep
if [ ! -f "$LOG/heads.done" ]; then
  step "4/8 head baselines sweep"
  ./run_all_heads.sh "$LTM_CKPT" > "$LOG/heads.log" 2>&1 && touch "$LOG/heads.done" || fail "heads sweep"
fi

# 5. partial plot
step "5/8 plotting (partial: LTM-only + heads)"
$PY plot_comparison.py > "$LOG/plot_partial.log" 2>&1 || step "plot (partial) had errors, continuing"

# 6. STM pretraining
if [ ! -f "$STM_CKPT" ]; then
  step "6/8 STM pretraining, 12 epochs"
  $PY cifar_main_stm_training.py --experiment-type pretrain --fine-classes 1 2 --coarse-classes $CC --epochs 12 \
    > "$LOG/stm_pretrain.log" 2>&1 || fail "STM pretraining"
fi

# 7. STM continual, 3 orders (sequential)
for order in "3 4 5" "4 5 3" "5 3 4"; do
  tag=$(echo "$order" | tr -d ' ')
  if [ ! -f "$LOG/stm_continual_$tag.done" ]; then
    step "7/8 STM continual, order $order"
    $PY cifar_main_stm_training.py --experiment-type continual --fine-classes $order --coarse-classes $CC \
      > "$LOG/stm_continual_$tag.log" 2>&1 && touch "$LOG/stm_continual_$tag.done" || fail "STM continual $order"
  fi
done

# 8. final plot
step "8/8 plotting (final)"
$PY plot_comparison.py > "$LOG/plot_final.log" 2>&1 || step "plot (final) had errors"
step "ALL DONE"
