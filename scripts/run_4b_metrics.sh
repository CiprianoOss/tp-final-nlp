#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="$PWD/src"

mkdir -p outputs/logs
mkdir -p outputs/evaluation
mkdir -p models/steering

MODEL="Qwen/Qwen3-4B-Instruct-2507"
ADAPTER="fireball-nlp/fireball-qwen3-4b-lora-10k"
STEERING="models/steering/fantasy_direction_4b.pt"
PROMPTS="data/evaluation/adversarial_prompts.jsonl"
CAMPAIGNS="data/evaluation/campaigns_50turn.jsonl"

echo "========== 1. Building adversarial set if needed =========="
python -m fireball_narrator.evaluation.build_adversarial_set

echo "========== 2. Checking steering vector =========="
test -f "$STEERING"
echo "Steering listo en $STEERING"

echo "========== 3. Evaluating 4B LoRA base: TAR/SMI =========="
python -m fireball_narrator.evaluation.evaluate_adversarial \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --prompts "$PROMPTS" \
  --output outputs/evaluation/4b_lora_theme.json \
  --load-in-4bit

echo "========== 4. Evaluating 4B LoRA + steering: TAR/SMI alpha 2,4,8,12 =========="
for ALPHA in 2 4 8 12; do
  echo "---- alpha=$ALPHA ----"
  python -m fireball_narrator.evaluation.evaluate_adversarial \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --prompts "$PROMPTS" \
    --steering-vector "$STEERING" \
    --alpha "$ALPHA" \
    --output "outputs/evaluation/4b_steering_alpha${ALPHA}_theme.json" \
    --load-in-4bit
done

echo "========== 5. Preparing campaign file =========="
if [ ! -f "$CAMPAIGNS" ]; then
  echo "No existe $CAMPAIGNS. Copiando campaigns.example.jsonl como prueba técnica."
  cp data/evaluation/campaigns.example.jsonl "$CAMPAIGNS"
fi

echo "========== 6. Memory evaluation: 4B LoRA + original history =========="
python -m fireball_narrator.evaluation.evaluate_retention \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --campaigns "$CAMPAIGNS" \
  --history-mode original \
  --output outputs/evaluation/4b_lora_original.json \
  --load-in-4bit

echo "========== 7. Memory evaluation: 4B LoRA + Caveman =========="
python -m fireball_narrator.evaluation.evaluate_retention \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --campaigns "$CAMPAIGNS" \
  --history-mode caveman \
  --output outputs/evaluation/4b_lora_caveman.json \
  --load-in-4bit

echo "========== 8. Memory evaluation: 4B LoRA + steering + original history =========="
python -m fireball_narrator.evaluation.evaluate_retention \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --campaigns "$CAMPAIGNS" \
  --history-mode original \
  --steering-vector "$STEERING" \
  --alpha 8 \
  --output outputs/evaluation/4b_steering_original.json \
  --load-in-4bit

echo "========== 9. Memory evaluation: 4B LoRA + Caveman + steering =========="
python -m fireball_narrator.evaluation.evaluate_retention \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --campaigns "$CAMPAIGNS" \
  --history-mode caveman \
  --steering-vector "$STEERING" \
  --alpha 8 \
  --output outputs/evaluation/4b_full_pipeline.json \
  --load-in-4bit

echo "========== DONE =========="
echo "Resultados en outputs/evaluation/"
ls -lh outputs/evaluation/4b_*.json
