from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from fireball_narrator.data.formatting import SYSTEM_PROMPT
from fireball_narrator.modeling import load_model_and_tokenizer, model_input_device
from fireball_narrator.steering.runtime import load_steering_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect last-token activations at every decoder layer."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--steering-vector", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-activations", type=Path)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
    )
    vectors = (
        load_steering_vectors(args.steering_vector)
        if args.steering_vector
        else {}
    )
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model_input_device(model)) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    rows = []
    saved = {}
    for layer_index, hidden_states in enumerate(outputs.hidden_states[1:]):
        activation = hidden_states[0, -1].float().cpu()
        row = {
            "layer": layer_index,
            "l2_norm": float(torch.linalg.vector_norm(activation)),
            "mean": float(activation.mean()),
            "std": float(activation.std()),
        }
        if layer_index in vectors:
            row["fantasy_direction_cosine"] = float(
                functional.cosine_similarity(
                    activation,
                    vectors[layer_index],
                    dim=0,
                )
            )
        rows.append(row)
        saved[str(layer_index)] = activation

    text = json.dumps(rows, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if args.save_activations:
        args.save_activations.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, args.save_activations)


if __name__ == "__main__":
    main()

