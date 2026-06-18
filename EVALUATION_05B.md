# Evaluación completa de Qwen 0.5B

Ejecutar todos los comandos desde la raíz del repositorio después de activar el
entorno virtual. Los comandos asumen que el adapter está en:

```text
outputs/qwen2.5-0.5b-fireball-qlora/adapter/
```

## 1. Preparar evaluación temática

```bash
git pull
pip install -r requirements.txt
python -m fireball_narrator.evaluation.build_adversarial_set
```

## 2. Medir el LoRA actual

TAR y SMI sin steering:

```bash
python -m fireball_narrator.evaluation.evaluate_adversarial \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/qwen2.5-0.5b-fireball-qlora/adapter \
  --prompts data/evaluation/adversarial_prompts.jsonl \
  --output outputs/evaluation/05b_lora_theme.json \
  --load-in-4bit
```

## 3. Construir steering

```bash
python -m fireball_narrator.steering.build_vector \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/qwen2.5-0.5b-fireball-qlora/adapter \
  --pairs data/steering/contrast_pairs.jsonl \
  --layers 6,12,18 \
  --output models/steering/fantasy_direction_05b.pt \
  --load-in-4bit
```

## 4. Medir LoRA más steering

```bash
python -m fireball_narrator.evaluation.evaluate_adversarial \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/qwen2.5-0.5b-fireball-qlora/adapter \
  --prompts data/evaluation/adversarial_prompts.jsonl \
  --steering-vector models/steering/fantasy_direction_05b.pt \
  --alpha 8 \
  --output outputs/evaluation/05b_steering_theme.json \
  --load-in-4bit
```

Comparar `TAR` y `SMI` de los dos JSON. Para el informe final, repetir steering
con `alpha` 2, 4, 8 y 12 y reportar cuál conserva mejor la calidad narrativa.

Sin `--judge-model`, TAR usa la heurística rápida incluida en el proyecto. Para
la medición final del trabajo, iniciar Ollama y agregar el mismo juez externo a
las dos ejecuciones temáticas:

```bash
--judge-model qwen3:8b
```

El modelo juez debe ser distinto del Qwen2.5-0.5B evaluado.

## 5. Preparar campañas para memoria

`data/evaluation/campaigns.example.jsonl` muestra el formato. Crear
`data/evaluation/campaigns_50turn.jsonl` con una línea JSON por campaña:

- `history`: lista de 50 turnos.
- `facts`: hechos y keywords que deben sobrevivir a la compresión.
- `qa`: preguntas cerradas con una o más respuestas aceptadas.

El archivo de ejemplo sirve para una prueba técnica, pero no para reportar
resultados académicos.

## 6. Ejecutar las cuatro condiciones de memoria

### A. LoRA con history original

```bash
python -m fireball_narrator.evaluation.evaluate_retention \
  --adapter outputs/qwen2.5-0.5b-fireball-qlora/adapter \
  --campaigns data/evaluation/campaigns_50turn.jsonl \
  --history-mode original \
  --output outputs/evaluation/05b_lora_original.json \
  --load-in-4bit
```

### B. LoRA con Caveman

```bash
python -m fireball_narrator.evaluation.evaluate_retention \
  --adapter outputs/qwen2.5-0.5b-fireball-qlora/adapter \
  --campaigns data/evaluation/campaigns_50turn.jsonl \
  --history-mode caveman \
  --output outputs/evaluation/05b_lora_caveman.json \
  --load-in-4bit
```

### C. LoRA con steering e history original

```bash
python -m fireball_narrator.evaluation.evaluate_retention \
  --adapter outputs/qwen2.5-0.5b-fireball-qlora/adapter \
  --campaigns data/evaluation/campaigns_50turn.jsonl \
  --history-mode original \
  --steering-vector models/steering/fantasy_direction_05b.pt \
  --alpha 8 \
  --output outputs/evaluation/05b_steering_original.json \
  --load-in-4bit
```

### D. Pipeline completo: LoRA, Caveman y steering

```bash
python -m fireball_narrator.evaluation.evaluate_retention \
  --adapter outputs/qwen2.5-0.5b-fireball-qlora/adapter \
  --campaigns data/evaluation/campaigns_50turn.jsonl \
  --history-mode caveman \
  --steering-vector models/steering/fantasy_direction_05b.pt \
  --alpha 8 \
  --output outputs/evaluation/05b_full_pipeline.json \
  --load-in-4bit
```

Cada reporte contiene:

- `TCR`: reducción de tokens aplicada por Caveman.
- `SPS`: proporción de hechos cuyas keywords siguen en el history usado.
- `IRA`: proporción de preguntas QA contestadas correctamente.

La comparación principal del PDF es maximizar TCR manteniendo IRA mayor o
igual a 0.90. TAR/SMI se interpretan por separado porque los ataques temáticos
no contienen history logs que Caveman pueda comprimir.
