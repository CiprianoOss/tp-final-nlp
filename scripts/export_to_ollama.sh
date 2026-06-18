#!/usr/bin/env bash
set -euo pipefail

MERGED_DIR="${1:-outputs/qwen3-4b-fireball-merged}"
GGUF_DIR="${2:-outputs/gguf}"
MODEL_SLUG="${3:-fireball-qwen3-4b}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-third_party/llama.cpp}"
MODEL_NAME="${OLLAMA_MODEL_NAME:-fireball-narrator}"

mkdir -p "$GGUF_DIR" third_party

if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
fi

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  "$MERGED_DIR" \
  --outfile "$GGUF_DIR/$MODEL_SLUG-f16.gguf" \
  --outtype f16

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DGGML_CUDA=ON
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j

"$LLAMA_CPP_DIR/build/bin/llama-quantize" \
  "$GGUF_DIR/$MODEL_SLUG-f16.gguf" \
  "$GGUF_DIR/$MODEL_SLUG-q4_k_m.gguf" \
  Q4_K_M

OLLAMA_GGUF="$(cd "$GGUF_DIR" && pwd)/$MODEL_SLUG-q4_k_m.gguf"
sed "s|__GGUF_PATH__|$OLLAMA_GGUF|g" ollama/Modelfile.template > "$GGUF_DIR/Modelfile"
ollama create "$MODEL_NAME" -f "$GGUF_DIR/Modelfile"
echo "Created Ollama model: $MODEL_NAME"
