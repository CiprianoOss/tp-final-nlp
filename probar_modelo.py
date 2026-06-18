from __future__ import annotations

"""
Demo interactiva más estable para Fireball Narrator.

Diferencia contra la versión anterior:
- El usuario escribe normal.
- Caveman se aplica SOLO sobre memoria factual de acciones del usuario.
- No metemos respuestas enteras del modelo en la memoria, porque eso hacía que
  el modelo copiara/reutilizara frases malas y entrara en loops.
- La generación es greedy por defecto y con controles anti-repetición.
- Usa Qwen base + LoRA + activation steering + constrained decoding.

Uso recomendado:

    python probar_modelo.py --load-in-4bit --alpha 2

Comandos dentro del chat:
    salir       termina
    limpiar     borra memoria
    debug       muestra memoria y prompt interno
    personaje   cambia personaje
"""

import argparse
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from fireball_narrator.compression.caveman import compress_history
from fireball_narrator.data.formatting import SYSTEM_PROMPT
from fireball_narrator.decoding import build_bad_words_ids, read_forbidden_terms
from fireball_narrator.steering.runtime import apply_steering, load_steering_vectors


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_MODEL = "fireball-nlp/fireball-qwen2.5-0.5b-lora"
DEFAULT_STEERING_PATH = Path("models/steering/fantasy_direction_05b.pt")
DEFAULT_BAD_WORDS_FILE = Path("data/evaluation/modern_terms.txt")


DEMO_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\nYou are not a chatbot assistant. Do not mention being Qwen or an AI. "
    + "Write concise fantasy narration only. Do not copy memory notes verbatim. "
    + "Avoid repetition. Output 1 or 2 short paragraphs at most."
)


def modelo_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def resolver_steering_path(path: Path, repo_id: str) -> Path:
    if path.exists():
        return path

    print(f"No encontré steering local en: {path}")
    print("Intentando descargar steering/fantasy_direction_05b.pt desde Hugging Face...")

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename="steering/fantasy_direction_05b.pt",
            repo_type="model",
        )
    except Exception as exc:
        raise FileNotFoundError(
            "No pude encontrar ni descargar el vector de steering.\n"
            "Solución manual:\n"
            "  mkdir -p models/steering\n"
            "  hf download fireball-nlp/fireball-qwen2.5-0.5b-lora "
            "steering/fantasy_direction_05b.pt --local-dir .\n"
            "  cp steering/fantasy_direction_05b.pt models/steering/fantasy_direction_05b.pt\n"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(downloaded).read_bytes())
    return path


def cargar_modelo(load_in_4bit: bool):
    print("Cargando Qwen base + LoRA adapter...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map: str | dict[str, str] = "auto"
        device_name = torch.cuda.get_device_name(0)

        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
    else:
        dtype = torch.float32
        device_map = {"": "cpu"}
        device_name = "CPU"
        if load_in_4bit:
            raise RuntimeError("--load-in-4bit requiere CUDA")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL)
    model.eval()

    print(f"Modelo listo en {device_name}.")
    return model, tokenizer


def normalizar_hecho_usuario(mensaje: str) -> str:
    """
    Guardamos acciones del usuario como hechos, no como diálogo a imitar.
    """
    msg = re.sub(r"\s+", " ", mensaje.strip())
    msg = msg.rstrip(".")
    return f"Player action/fact: {msg}."


def construir_prompt(
    mensaje_usuario: str,
    memoria_factos: list[str],
    personaje: dict[str, str],
    max_memory_items: int,
) -> tuple[str, list[str]]:
    memoria_reciente = memoria_factos[-max_memory_items:]
    memoria_caveman = compress_history(memoria_reciente)

    if memoria_caveman:
        memoria_txt = "\n".join(f"- {item}" for item in memoria_caveman)
    else:
        memoria_txt = "None yet."

    nombre = personaje["name"]
    raza = personaje["race"]
    clase = personaje["class"]
    hp = personaje["hp"]

    prompt = f"""Compressed factual campaign memory:
{memoria_txt}

Important instruction:
The compressed memory above is only factual notes. Do not quote it directly. Do not repeat note fragments.
Use it only to preserve continuity.

Current character:
Name: {nombre}
Race: {raza}
Class: {clase}
HP: {hp}
Inventory: lantern, rusty dagger, small pouch of silver coins

Current player action:
{mensaje_usuario}

Narration task:
Write the next medieval-fantasy narrative turn in 1 or 2 concise paragraphs.
React to the current action. Preserve relevant memory. Do not introduce modern technology or science-fiction elements.
If the player mentions a modern object, reject it or reinterpret it as a strange in-world magical misunderstanding.
Do not repeat the same noun or phrase many times."""

    return prompt, memoria_caveman


def limpiar_salida(texto: str, max_sentences: int = 3) -> str:
    """
    Corte defensivo para demos: si el modelo empieza a repetir, nos quedamos
    con las primeras oraciones coherentes.
    """
    texto = texto.strip()
    texto = re.sub(r"\s+", " ", texto)

    # Cortar loops obvios de palabras repetidas.
    texto = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", texto, flags=re.IGNORECASE)

    # Cortar repeticiones obvias de frase corta.
    chunks = re.split(r"(?<=[.!?])\s+", texto)
    kept: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        cleaned = chunk.strip()
        if not cleaned:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        kept.append(cleaned)
        if len(kept) >= max_sentences:
            break

    resultado = " ".join(kept).strip()
    return resultado or texto[:500].strip()


def generar(
    model: Any,
    tokenizer: Any,
    prompt: str,
    steering_vectors: Any | None,
    alpha: float,
    bad_words_ids: list[list[int]] | None,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": DEMO_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(modelo_input_device(model)) for k, v in inputs.items()}

    steering_context = (
        apply_steering(model, steering_vectors, alpha)
        if steering_vectors is not None
        else nullcontext()
    )

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.25,
        no_repeat_ngram_size=4,
        bad_words_ids=bad_words_ids,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.inference_mode(), steering_context:
        output = model.generate(**gen_kwargs)

    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return limpiar_salida(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo Fireball Qwen 0.5B con LoRA + Caveman + steering.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-memory-items", type=int, default=20)
    parser.add_argument("--no-steering", action="store_true")
    parser.add_argument("--no-constrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model, tokenizer = cargar_modelo(args.load_in_4bit)

    steering_vectors = None
    if not args.no_steering:
        steering_path = resolver_steering_path(DEFAULT_STEERING_PATH, LORA_MODEL)
        steering_vectors = load_steering_vectors(steering_path)
        print(f"Activation steering: ON | alpha={args.alpha} | path={steering_path}")
    else:
        print("Activation steering: OFF")

    bad_words_ids = None
    if not args.no_constrained and DEFAULT_BAD_WORDS_FILE.exists():
        terms = read_forbidden_terms(DEFAULT_BAD_WORDS_FILE)
        bad_words_ids = build_bad_words_ids(tokenizer, terms)
        print(f"Constrained decoding: ON | terms={len(terms)}")
    else:
        print("Constrained decoding: OFF")

    personaje = {"name": "Pip", "race": "Gnome", "class": "Rogue", "hp": "12"}
    memoria_factos: list[str] = []

    ultimo_prompt = ""
    ultima_memoria = []

    print("\nPipeline demo activo.")
    print("Usuario escribe normal; Caveman comprime memoria factual interna.")
    print("Comandos: salir | limpiar | debug | personaje")
    print("\nProbá: I enter the cave and look around for tracks.\n")

    while True:
        try:
            mensaje = input("\nVos: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not mensaje:
            continue

        cmd = mensaje.casefold()

        if cmd in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break

        if cmd == "limpiar":
            memoria_factos.clear()
            print("Memoria reiniciada.")
            continue

        if cmd == "personaje":
            personaje["name"] = input("Nombre [Pip]: ").strip() or "Pip"
            personaje["race"] = input("Raza [Gnome]: ").strip() or "Gnome"
            personaje["class"] = input("Clase [Rogue]: ").strip() or "Rogue"
            personaje["hp"] = input("HP [12]: ").strip() or "12"
            print(f"Personaje: {personaje['name']} ({personaje['race']} {personaje['class']}) HP {personaje['hp']}")
            continue

        if cmd == "debug":
            print("\n--- MEMORIA CAVEMAN ---")
            if ultima_memoria:
                for item in ultima_memoria:
                    print(f"- {item}")
            else:
                print("None yet.")
            print("\n--- PROMPT INTERNO ---")
            print(ultimo_prompt or "Todavía no hay prompt.")
            continue

        prompt, memoria_caveman = construir_prompt(
            mensaje_usuario=mensaje,
            memoria_factos=memoria_factos,
            personaje=personaje,
            max_memory_items=args.max_memory_items,
        )

        respuesta = generar(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steering_vectors=steering_vectors,
            alpha=args.alpha,
            bad_words_ids=bad_words_ids,
            max_new_tokens=args.max_new_tokens,
        )

        print(f"\nModelo: {respuesta}")

        memoria_factos.append(normalizar_hecho_usuario(mensaje))
        # Guardamos respuesta resumida como hecho suave, pero no la respuesta entera.
        # Esto evita que el modelo copie sus propios errores en turnos futuros.
        if respuesta:
            memoria_factos.append(f"Narrator outcome summary: {respuesta[:180]}")

        ultimo_prompt = prompt
        ultima_memoria = memoria_caveman


if __name__ == "__main__":
    main()
