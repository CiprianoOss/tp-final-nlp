# FIREBALL Narrative Agent

Pipeline reproducible para ajustar un Qwen abierto con FIREBALL, estudiar sus
capas, aplicar activation steering temático y comprimir history logs.

## Decisiones técnicas

- Modelo por defecto: `Qwen/Qwen3-4B-Instruct-2507`.
- Entrenamiento: QLoRA de 4 bits con `transformers`, PEFT y bitsandbytes.
- Tarea FIREBALL: `state-to-narration`, siguiendo la preparación usada por el
  trabajo original. El input contiene historia, estado, actor, objetivos,
  comando y resultado; el target es `after_utterances`.
- Steering: se calcula una dirección
  `fantasía medieval - ciencia ficción` y se inyecta mediante forward hooks en
  capas residuales seleccionadas.
- Constrained decoding: opcionalmente se bloquean secuencias modernas en los
  logits con `bad_words_ids`.
- Ollama: se usa al final para inferencia. El adapter se fusiona con el modelo,
  se convierte a GGUF y se importa en Ollama.

Esto es transfer learning/post-training, no pretraining desde cero. Ollama no
es el framework de entrenamiento ni permite observar activaciones internas.
Además, su soporte directo de adapters Safetensors no incluye Qwen en la
documentación actual; por eso la ruta confiable es `LoRA -> merge -> GGUF`.

## Estructura

```text
configs/                  hiperparámetros QLoRA
data/processed/           splits SFT generados
data/steering/            pares contrastivos
models/steering/          vectores de dirección
ollama/                   plantilla Modelfile
outputs/                  checkpoints, modelos y evaluaciones
scripts/                  entrenamiento y exportación
src/fireball_narrator/
  data/                   descarga y formato FIREBALL
  training/               QLoRA y merge
  steering/               vector, hooks e inspección
  compression/            Caveman logging heurístico
  evaluation/             TAR, SMI, TCR, SPS e IRA
tests/                    pruebas sin GPU
```

## Requisitos

Recomendado para la máquina con GPU:

- Linux o WSL2.
- Python 3.10 o 3.11.
- NVIDIA GPU y drivers CUDA recientes.
- Git, CMake y compilador C++ para exportar GGUF.
- Ollama solo para la etapa final.

El QLoRA de 4B suele necesitar aproximadamente 10-14 GB de VRAM con secuencias
de 2048 tokens. Si falta memoria, reducir `max_seq_length` a 1024 o subir
`gradient_accumulation_steps`. Las cifras dependen de CUDA, batch y versiones.

También se incluye una variante de 0.5B basada en
`Qwen/Qwen2.5-0.5B-Instruct`. Con QLoRA suele entrar cómodamente en una GPU de
6 GB; si falta memoria, bajar los batch sizes de 4 a 1 o 2.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si la versión de PyTorch instalada no coincide con CUDA, instalar primero el
wheel indicado por [PyTorch](https://pytorch.org/get-started/locally/) y luego
ejecutar `pip install -r requirements.txt`.

FIREBALL y Qwen son públicos, pero autenticar Hugging Face evita límites bajos:

```bash
huggingface-cli login
```

## 1. Preparar FIREBALL

La preparación divide por archivo de sesión, no por fila. Así no aparecen
turnos de la misma partida en train y evaluación.

Prueba corta:

```bash
python -m fireball_narrator.data.prepare_fireball \
  --output-dir data/processed \
  --max-train-samples 2000 \
  --max-validation-samples 200 \
  --max-test-samples 200
```

Preparación completa:

```bash
python -m fireball_narrator.data.prepare_fireball \
  --output-dir data/processed
```

La descarga completa de Hugging Face ocupa cerca de 2 GB. También puede usarse
una copia ya descargada:

```bash
python -m fireball_narrator.data.prepare_fireball \
  --local-data-dir /ruta/a/FIREBALL/filtered \
  --output-dir data/processed
```

Archivos resultantes:

- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`
- `data/processed/manifest.json`

## 2. Entrenar QLoRA

Primero hacer una corrida corta cambiando temporalmente en el YAML:

```yaml
training:
  num_train_epochs: 0.05
```

Entrenamiento normal:

```bash
bash scripts/train.sh configs/qwen3_4b_qlora.yaml
```

Equivalente explícito:

```bash
accelerate launch -m fireball_narrator.training.train_qlora \
  --config configs/qwen3_4b_qlora.yaml
```

El adapter queda en:

```text
outputs/qwen3-4b-fireball-qlora/adapter/
```

Para continuar un checkpoint:

```bash
accelerate launch -m fireball_narrator.training.train_qlora \
  --config configs/qwen3_4b_qlora.yaml \
  --resume-from-checkpoint outputs/qwen3-4b-fireball-qlora/checkpoint-XXXX
```

### Variante Qwen 0.5B

Usa los mismos splits FIREBALL ya preparados. No hace falta volver a descargar
ni procesar el dataset:

```bash
bash scripts/train_05b.sh
```

Equivalente explícito:

```bash
accelerate launch -m fireball_narrator.training.train_qlora \
  --config configs/qwen2_5_05b_qlora.yaml
```

Esta configuración usa `Qwen/Qwen2.5-0.5B-Instruct`, LoRA rank 16, batch
efectivo 16 y tres épocas. El adapter queda en:

```text
outputs/qwen2.5-0.5b-fireball-qlora/adapter/
```

## 3. Fusionar y exportar a Ollama

Fusionar LoRA con el modelo base:

```bash
python -m fireball_narrator.training.merge_adapter \
  --adapter outputs/qwen3-4b-fireball-qlora/adapter \
  --output outputs/qwen3-4b-fireball-merged \
  --dtype bfloat16
```

Convertir, cuantizar a `Q4_K_M` y crear el modelo Ollama:

```bash
bash scripts/export_to_ollama.sh \
  outputs/qwen3-4b-fireball-merged \
  outputs/gguf
```

Para fusionar y exportar el modelo de 0.5B:

```bash
python -m fireball_narrator.training.merge_adapter \
  --adapter outputs/qwen2.5-0.5b-fireball-qlora/adapter \
  --output outputs/qwen2.5-0.5b-fireball-merged \
  --dtype bfloat16

OLLAMA_MODEL_NAME=fireball-narrator-05b \
bash scripts/export_to_ollama.sh \
  outputs/qwen2.5-0.5b-fireball-merged \
  outputs/gguf \
  fireball-qwen2.5-0.5b
```

Qwen2.5-0.5B tiene 24 capas. Para steering, empezar con las capas `6,12,18`:

```bash
python -m fireball_narrator.steering.build_vector \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --adapter outputs/qwen2.5-0.5b-fireball-qlora/adapter \
  --layers 6,12,18 \
  --output models/steering/fantasy_direction_05b.pt \
  --load-in-4bit
```

El script descarga `llama.cpp`, compila con CUDA, produce el GGUF y ejecuta:

```bash
ollama create fireball-narrator -f outputs/gguf/Modelfile
```

Prueba:

```bash
ollama run fireball-narrator
```

## 4. Construir el vector de steering

Los pares contrastivos están en `data/steering/contrast_pairs.jsonl`.

```bash
python -m fireball_narrator.steering.build_vector \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter outputs/qwen3-4b-fireball-qlora/adapter \
  --pairs data/steering/contrast_pairs.jsonl \
  --layers 12,18,24 \
  --output models/steering/fantasy_direction.pt \
  --load-in-4bit
```

Los índices `12,18,24` son un punto inicial, no un resultado experimental. Se
deben comparar capas e intensidades con TAR y calidad narrativa.

## 5. Inspeccionar capas y generar

Resumen de activaciones de todas las capas:

```bash
python -m fireball_narrator.steering.inspect_layers \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter outputs/qwen3-4b-fireball-qlora/adapter \
  --steering-vector models/steering/fantasy_direction.pt \
  --prompt "I activate my jetpack and fly over the castle." \
  --output outputs/layer_inspection.json \
  --save-activations outputs/layer_activations.pt \
  --load-in-4bit
```

Generación con steering:

```bash
python -m fireball_narrator.steering.generate \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter outputs/qwen3-4b-fireball-qlora/adapter \
  --steering-vector models/steering/fantasy_direction.pt \
  --alpha 8 \
  --prompt "I pull out a laser rifle and threaten the dragon." \
  --bad-words-file data/evaluation/modern_terms.txt \
  --load-in-4bit
```

Probar al menos `alpha=0,2,4,8,12`. Valores altos pueden degradar gramática o
repetir vocabulario de fantasía. El steering se ejecuta con Transformers, no
con Ollama, porque necesita acceso al forward pass.

## 6. Evaluación adversaria

Crear exactamente 100 prompts disruptivos:

```bash
python -m fireball_narrator.evaluation.build_adversarial_set
```

Evaluar TAR y SMI:

```bash
python -m fireball_narrator.evaluation.evaluate_adversarial \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter outputs/qwen3-4b-fireball-qlora/adapter \
  --prompts data/evaluation/adversarial_prompts.jsonl \
  --steering-vector models/steering/fantasy_direction.pt \
  --alpha 8 \
  --bad-words-file data/evaluation/modern_terms.txt \
  --output outputs/evaluation/steered.json \
  --load-in-4bit
```

Sin `--judge-model` se usa una heurística rápida. Para el resultado del trabajo,
usar un juez externo en Ollama:

```bash
python -m fireball_narrator.evaluation.evaluate_adversarial \
  ... \
  --judge-model qwen3:8b
```

Comparar al menos:

1. Qwen base.
2. Qwen + LoRA.
3. Qwen + LoRA + constrained decoding.
4. Qwen + LoRA + steering para varias capas y `alpha`.
5. Qwen + LoRA + steering + constrained decoding.

## 7. Compresión de history logs

El input JSONL debe contener un campo `history` con una lista de strings:

```bash
python -m fireball_narrator.compression.caveman \
  --input data/history/campaigns.jsonl \
  --output outputs/compressed_campaigns.jsonl \
  --tokenizer Qwen/Qwen3-4B-Instruct-2507
```

El output agrega `compressed_history` y `token_compression_ratio`. Para SPS e
IRA hacen falta facts y respuestas QA anotados; las funciones están en
`fireball_narrator.evaluation.metrics`.

## Validación local

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall src tests
```

La secuencia completa para comparar el LoRA de 0.5B antes y después de Caveman
y steering está en [EVALUATION_05B.md](EVALUATION_05B.md).

## Limitaciones del dataset

- FIREBALL está principalmente en inglés.
- No es un corpus puro de Dungeon Masters: contiene juego real de distintos
  participantes y comandos Avrae. El ajuste aprende continuación narrativa,
  pero no garantiza por sí solo una persona de DM.
- FIREBALL no contiene suficientes ataques modernos para entrenar rechazo de
  anacronismos. Esa propiedad se mide y refuerza con contrast pairs, steering y
  evaluación adversaria.
- Antes de afirmar mejoras hay que reportar varias seeds y comparar contra
  base y LoRA sin steering.

## Fuentes

- [FIREBALL en Hugging Face](https://huggingface.co/datasets/lara-martin/FIREBALL)
- [Repositorio FIREBALL](https://github.com/zhudotexe/FIREBALL)
- [Paper ACL 2023](https://aclanthology.org/2023.acl-long.229/)
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.mdx)
- [llama.cpp para Qwen](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
