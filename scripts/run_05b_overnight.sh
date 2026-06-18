#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate
export PYTHONPATH="$PWD/src"

mkdir -p outputs/logs
mkdir -p outputs/evaluation
mkdir -p models/steering

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER="outputs/qwen2.5-0.5b-fireball-qlora/adapter"
STEERING="models/steering/fantasy_direction_05b.pt"
PROMPTS="data/evaluation/adversarial_prompts.jsonl"
CAMPAIGNS="data/evaluation/campaigns_50turn.jsonl"

echo "========== 1. Building adversarial set =========="
python -m fireball_narrator.evaluation.build_adversarial_set

echo "========== 2. Training Qwen2.5-0.5B QLoRA =========="
bash scripts/train_05b.sh

echo "========== 3. Checking adapter =========="
test -f "$ADAPTER/adapter_model.safetensors"
echo "Adapter listo en $ADAPTER"

echo "========== 4. Evaluating LoRA base: TAR/SMI =========="
python -m fireball_narrator.evaluation.evaluate_adversarial \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --prompts "$PROMPTS" \
  --output outputs/evaluation/05b_lora_theme.json \
  --load-in-4bit

echo "========== 5. Building activation steering =========="
python -m fireball_narrator.steering.build_vector \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --pairs data/steering/contrast_pairs.jsonl \
  --layers 6,12,18 \
  --output "$STEERING" \
  --load-in-4bit

test -f "$STEERING"
echo "Steering listo en $STEERING"

echo "========== 6. Evaluating LoRA + steering for alpha 2,4,8,12 =========="
for ALPHA in 2 4 8 12; do
  echo "---- alpha=$ALPHA ----"
  python -m fireball_narrator.evaluation.evaluate_adversarial \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --prompts "$PROMPTS" \
    --steering-vector "$STEERING" \
    --alpha "$ALPHA" \
    --output "outputs/evaluation/05b_steering_alpha${ALPHA}_theme.json" \
    --load-in-4bit
done

echo "========== 7. Preparing campaign file =========="
if [ ! -f "$CAMPAIGNS" ]; then
  echo "No existe $CAMPAIGNS. Copiando campaigns.example.jsonl como prueba técnica."
  cp data/evaluation/campaigns.example.jsonl "$CAMPAIGNS"
fi

echo "========== 8. Memory evaluation: LoRA + original history =========="
python -m fireball_narrator.evaluation.evaluate_retention \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --campaigns "$CAMPAIGNS" \
  --history-mode original \
  --output outputs/evaluation/05b_lora_original.json \
  --load-in-4bit

echo "========== 9. Memory evaluation: LoRA + Caveman =========="
python -m fireball_narrator.evaluation.evaluate_retention \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --campaigns "$CAMPAIGNS" \
  --history-mode caveman \
  --output outputs/evaluation/05b_lora_caveman.json \
  --load-in-4bit

echo "========== 10. Memory evaluation: LoRA + steering + original history =========="
python -m fireball_narrator.evaluation.evaluate_retention \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --campaigns "$CAMPAIGNS" \
  --history-mode original \
  --steering-vector "$STEERING" \
  --alpha 8 \
  --output outputs/evaluation/05b_steering_original.json \
  --load-in-4bit

echo "========== 11. Memory evaluation: LoRA + Caveman + steering =========="
python -m fireball_narrator.evaluation.evaluate_retention \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --campaigns "$CAMPAIGNS" \
  --history-mode caveman \
  --steering-vector "$STEERING" \
  --alpha 8 \
  --output outputs/evaluation/05b_full_pipeline.json \
  --load-in-4bit

echo "========== DONE =========="
echo "Resultados en outputs/evaluation/"
ls -lh outputs/evaluation/
