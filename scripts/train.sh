#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/qwen3_4b_qlora.yaml}"
accelerate launch -m fireball_narrator.training.train_qlora --config "$CONFIG"

