from __future__ import annotations

"""
Demo 4B stateful para Fireball Narrator.

Pipeline:
- Qwen3-4B base
- LoRA 10k
- activation steering 4B
- Caveman sobre memoria factual
- estado duro separado del modelo
- validador simple de anacronismos antes de llamar al modelo

Uso recomendado:
    python probar_modelo_4b_stateful.py --load-in-4bit --alpha 4 --max-new-tokens 100

Comandos:
    salir
    limpiar
    debug
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
from fireball_narrator.decoding import build_bad_words_ids, read_forbidden_terms
from fireball_narrator.steering.runtime import apply_steering, load_steering_vectors


BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LORA_MODEL = "fireball-nlp/fireball-qwen3-4b-lora-10k"
DEFAULT_STEERING_PATH = Path("models/steering/fantasy_direction_4b.pt")
DEFAULT_BAD_WORDS_FILE = Path("data/evaluation/modern_terms.txt")


SYSTEM_PROMPT = """You are the narrator for a medieval fantasy RPG.
You are not a chatbot. Do not thank the player. Do not end the adventure unless the player explicitly leaves.
React directly to the player's command.
The hard game state is the source of truth. Do not contradict it.
Do not introduce modern technology or science-fiction.
Write concise, grounded narration in 1 short paragraph.
"""


ANACHRONISTIC_TERMS = [
    "smartphone",
    "phone",
    "flashlight",
    "spaceship",
    "space ship",
    "radio",
    "laser",
    "internet",
    "computer",
    "satellite",
    "scanner",
    "gps",
]


def model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def resolver_steering_path(path: Path, repo_id: str) -> Path:
    if path.exists():
        return path

    print(f"No encontré steering local en: {path}")
    print("Intentando descargar steering/fantasy_direction_4b.pt desde Hugging Face...")

    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename="steering/fantasy_direction_4b.pt",
        repo_type="model",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(downloaded).read_bytes())
    return path


def cargar_modelo(load_in_4bit: bool):
    print("Cargando Qwen3-4B base + LoRA 10k...")

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

    # Evita que config vieja de sampling ensucie la generación greedy.
    try:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    except Exception:
        pass

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Modelo listo en {device_name}.")
    return model, tokenizer


def nuevo_estado() -> dict[str, Any]:
    return {
        "character": {
            "name": "Pip",
            "race": "Gnome",
            "class": "Rogue",
            "hp": "12",
        },
        "location": "dark cave beneath the old hills",
        "inventory": ["lantern", "rusty dagger", "small pouch of silver coins"],
        "pouch": ["small pouch of silver coins"],
        "marks": [],
        "known_facts": [],
    }


def detectar_anacronismo(texto: str) -> str | None:
    lower = texto.casefold()
    for term in ANACHRONISTIC_TERMS:
        if term in lower:
            return term
    return None


def responder_anacronismo(term: str, state: dict[str, Any], user_msg: str) -> str:
    name = state["character"]["name"]
    location = state["location"]

    if term in {"smartphone", "phone", "flashlight"}:
        return (
            f"{name} reaches for the strange idea of a {term}, but no such device exists in this age of stone, "
            f"iron, and old magic. In the {location}, he instead raises his lantern and lets its trembling flame "
            "search the darkness."
        )

    if term in {"spaceship", "space ship", "radio", "satellite", "scanner", "gps"}:
        return (
            f"No distant machine answers {name}'s impossible call. The thought fades like a fever-dream, leaving only "
            f"the damp cave, the lantern light, and the weight of his rusty dagger in hand."
        )

    return (
        f"The notion of a {term} has no place in this medieval world. {name} steadies himself and relies instead on "
        "lantern light, caution, and old-fashioned courage."
    )


def es_pregunta_memoria(texto: str) -> bool:
    lower = texto.casefold()
    return (
        "what object" in lower
        or "what did i place" in lower
        or "what did i put" in lower
        or "what item" in lower
        or "qué objeto" in lower
        or "que objeto" in lower
    ) and ("pouch" in lower or "bolsa" in lower or "morral" in lower)


def responder_memoria(state: dict[str, Any]) -> str:
    pouch_items = [item for item in state["pouch"] if "pouch" not in item]
    if not pouch_items:
        return "Pip checks his pouch, but he has not placed any special object inside it yet."
    if len(pouch_items) == 1:
        return f"Pip remembers clearly: he placed {pouch_items[0]} inside his pouch."
    return "Pip remembers the objects in his pouch: " + ", ".join(pouch_items) + "."



def es_pregunta_marca(texto: str) -> bool:
    lower = texto.casefold()
    mark_words = [
        "what mark",
        "which mark",
        "how did i mark",
        "how i marked",
        "mark did i leave",
        "mark i left",
        "marked the tunnel",
        "marked the wall",
        "marca",
        "marqué",
        "marque",
        "señal",
        "senal",
    ]
    location_words = [
        "fork",
        "tunnel",
        "wall",
        "behind",
        "camino",
        "tunel",
        "túnel",
        "pared",
        "bifurcacion",
        "bifurcación",
    ]
    return any(w in lower for w in mark_words) and any(w in lower for w in location_words)


def responder_marca(state: dict[str, Any]) -> str:
    marks = state.get("marks", [])
    name = state["character"]["name"]

    if not marks:
        return f"{name} tries to remember, but he has not left any clear navigation mark yet."

    if len(marks) == 1:
        return f"{name} remembers clearly: he marked the way with {marks[0]}."

    return f"{name} remembers the marks he left: " + ", ".join(marks) + "."

def actualizar_estado_desde_usuario(state: dict[str, Any], texto: str) -> list[str]:
    lower = texto.casefold()
    facts = []

    if "blue cloth" in lower and "pouch" in lower:
        item = "a torn piece of blue cloth"
        if item not in state["pouch"]:
            state["pouch"].append(item)
        if item not in state["inventory"]:
            state["inventory"].append(item)
        facts.append("Pip placed a torn piece of blue cloth inside his pouch.")

    if "left tunnel" in lower:
        state["location"] = "left tunnel of the cave"
        facts.append("Pip chose the left tunnel.")

    if "mark" in lower and "wall" in lower and ("dagger" in lower or "scratch" in lower):
        mark = "a small dagger scratch on the wall near the fork"
        if mark not in state["marks"]:
            state["marks"].append(mark)
        facts.append("Pip marked the wall near the fork with a small dagger scratch.")

    if "fork" in lower and "tunnel" in lower:
        facts.append("Pip reached a fork in the cave tunnel.")

    if "water dripping" in lower:
        facts.append("Pip heard water dripping deeper in the cave.")

    if "tracks" in lower or "mud" in lower:
        facts.append("Pip examined mud and tracks inside the cave.")

    for fact in facts:
        if fact not in state["known_facts"]:
            state["known_facts"].append(fact)

    return facts


def normalizar_hecho_usuario(mensaje: str) -> str:
    msg = re.sub(r"\s+", " ", mensaje.strip()).rstrip(".")
    return f"Player action/fact: {msg}."


def construir_prompt(
    mensaje_usuario: str,
    memoria_factos: list[str],
    state: dict[str, Any],
    max_memory_items: int,
) -> tuple[str, list[str]]:
    memoria_caveman = compress_history(memoria_factos[-max_memory_items:])
    memoria_txt = "\n".join(f"- {item}" for item in memoria_caveman) if memoria_caveman else "None yet."

    char = state["character"]
    inventory = ", ".join(state["inventory"])
    pouch = ", ".join(state["pouch"])
    marks = ", ".join(state["marks"]) if state["marks"] else "none"

    hard_state = f"""Character: {char['name']} ({char['race']} {char['class']}), HP {char['hp']}
Location: {state['location']}
Inventory: {inventory}
Pouch contents: {pouch}
Navigation marks: {marks}"""

    prompt = f"""HARD GAME STATE:
{hard_state}

COMPRESSED HISTORY NOTES:
{memoria_txt}

RULES:
- The hard game state is exact and must be preserved.
- The player is currently in: {state['location']}.
- Do not move the character out of the current location unless the player explicitly does so.
- Do not invent merchants, payment, prize money, towns, or completed quests unless present in the hard state.
- Do not thank the player.
- React to the current action, not to a different imagined action.

CURRENT PLAYER ACTION:
{mensaje_usuario}

TASK:
Narrate the immediate result of the current action in one concise medieval-fantasy paragraph."""
    return prompt, memoria_caveman


def limpiar_salida(texto: str, max_sentences: int = 3) -> str:
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL | re.IGNORECASE)
    texto = texto.strip()
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"\b(\w+)(?:\s+\1\b){2,}", r"\1", texto, flags=re.IGNORECASE)

    sentences = re.split(r"(?<=[.!?])\s+", texto)
    kept = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        kept.append(s)
        if len(kept) >= max_sentences:
            break
    return " ".join(kept).strip() or texto[:500].strip()


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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(model_input_device(model)) for k, v in inputs.items()}

    steering_context = apply_steering(model, steering_vectors, alpha) if steering_vectors else nullcontext()

    with torch.inference_mode(), steering_context:
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.18,
            no_repeat_ngram_size=4,
            bad_words_ids=bad_words_ids,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return limpiar_salida(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo 4B stateful con LoRA + Caveman + steering.")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--max-memory-items", type=int, default=30)
    parser.add_argument("--no-steering", action="store_true")
    parser.add_argument("--no-constrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = nuevo_estado()
    memoria_factos: list[str] = []
    ultimo_prompt = ""
    ultima_memoria: list[str] = []

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

    print("\nDemo 4B stateful activa.")
    print("Comandos: salir | limpiar | debug")
    print("Primero probá pocos turnos y luego memoria/anacronismos.\n")

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
            state = nuevo_estado()
            ultimo_prompt = ""
            ultima_memoria = []
            print("Memoria y estado reiniciados.")
            continue

        if cmd == "debug":
            print("\n--- HARD STATE ---")
            print(f"Location: {state['location']}")
            print(f"Inventory: {', '.join(state['inventory'])}")
            print(f"Pouch: {', '.join(state['pouch'])}")
            print(f"Marks: {', '.join(state['marks']) if state['marks'] else 'none'}")
            print("\n--- MEMORIA CAVEMAN ---")
            if ultima_memoria:
                for item in ultima_memoria:
                    print(f"- {item}")
            else:
                print("None yet.")
            print("\n--- PROMPT INTERNO ---")
            print(ultimo_prompt or "Todavía no hay prompt.")
            continue

        # 1) Estado duro se actualiza antes.
        nuevos_hechos = actualizar_estado_desde_usuario(state, mensaje)
        memoria_factos.append(normalizar_hecho_usuario(mensaje))
        for fact in nuevos_hechos:
            memoria_factos.append(f"Hard-state fact: {fact}")

        # 2) Preguntas de memoria se responden desde estado duro.
        if es_pregunta_memoria(mensaje):
            respuesta = responder_memoria(state)
            print(f"\nModelo: {respuesta}")
            memoria_factos.append(f"Memory question answered from hard state: {respuesta}")
            continue

        # 3) Preguntas sobre marcas/camino se responden desde estado duro.
        if es_pregunta_marca(mensaje):
            respuesta = responder_marca(state)
            print(f"\nModelo: {respuesta}")
            memoria_factos.append(f"Navigation mark question answered from hard state: {respuesta}")
            continue

        # 4) Acciones anacrónicas se validan antes de llamar al modelo.
        invalid_term = detectar_anacronismo(mensaje)
        if invalid_term:
            respuesta = responder_anacronismo(invalid_term, state, mensaje)
            print(f"\nModelo: {respuesta}")
            memoria_factos.append(
                f"Invalid anachronistic attempt rejected/reinterpreted: player mentioned {invalid_term}."
            )
            continue

        # 5) Acción válida: usar LLM con Caveman + hard state.
        prompt, memoria_caveman = construir_prompt(
            mensaje_usuario=mensaje,
            memoria_factos=memoria_factos,
            state=state,
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

        # No guardamos la respuesta completa del modelo para no contaminar memoria.
        ultimo_prompt = prompt
        ultima_memoria = memoria_caveman


if __name__ == "__main__":
    main()
