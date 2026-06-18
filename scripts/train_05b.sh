#!/usr/bin/env bash
set -euo pipefail

accelerate launch -m fireball_narrator.training.train_qlora \
  --config configs/qwen2_5_05b_qlora.yaml

