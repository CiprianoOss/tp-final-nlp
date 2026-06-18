#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="outputs/qwen2.5-0.5b-fireball-qlora"

if [ -d "$OUTPUT_DIR" ]; then
  LAST_CKPT=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sort -V | tail -1)
else
  LAST_CKPT=""
fi

if [ -n "$LAST_CKPT" ]; then
  echo "Resuming from checkpoint: $LAST_CKPT"
  accelerate launch -m fireball_narrator.training.train_qlora \
    --config configs/qwen2_5_05b_qlora.yaml \
    --resume-from-checkpoint "$LAST_CKPT"
else
  echo "No checkpoint found. Starting from scratch."
  accelerate launch -m fireball_narrator.training.train_qlora \
    --config configs/qwen2_5_05b_qlora.yaml
fi
