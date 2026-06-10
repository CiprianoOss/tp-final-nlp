from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from fireball_narrator.modeling import load_model_and_tokenizer, model_input_device

PAIR_SYSTEM_PROMPT = (
    "Read the following short scene and represent its genre and world setting."
)


def read_pairs(path: Path) -> list[dict[str, str]]:
    pairs = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            pair = json.loads(line)
            if not pair.get("positive") or not pair.get("negative"):
                raise ValueError(f"Missing positive/negative at {path}:{line_number}")
            pairs.append(pair)
    if not pairs:
        raise ValueError(f"No contrast pairs found in {path}")
    return pairs


def encode_hidden_states(model, tokenizer, text: str) -> tuple[torch.Tensor, ...]:
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": PAIR_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {key: value.to(model_input_device(model)) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    return outputs.hidden_states


def parse_layers(value: str) -> list[int]:
    layers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not layers:
        raise ValueError("At least one layer is required")
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fantasy-minus-science-fiction activation directions."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/steering/contrast_pairs.jsonl"),
    )
    parser.add_argument("--layers", default="12,18,24")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/steering/fantasy_direction.pt"),
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_indices = parse_layers(args.layers)
    pairs = read_pairs(args.pairs)
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
    )

    layer_count = int(model.config.num_hidden_layers)
    for layer_index in layer_indices:
        if layer_index < 0 or layer_index >= layer_count:
            raise IndexError(
                f"Layer {layer_index} is outside the model range 0..{layer_count - 1}"
            )

    sums: dict[int, torch.Tensor] = {}
    for pair in pairs:
        positive = encode_hidden_states(model, tokenizer, pair["positive"])
        negative = encode_hidden_states(model, tokenizer, pair["negative"])
        for layer_index in layer_indices:
            difference = (
                positive[layer_index + 1][0, -1].float().cpu()
                - negative[layer_index + 1][0, -1].float().cpu()
            )
            sums[layer_index] = sums.get(layer_index, torch.zeros_like(difference))
            sums[layer_index] += difference

    vectors = {
        str(layer_index): functional.normalize(
            total / len(pairs),
            dim=0,
        )
        for layer_index, total in sums.items()
    }
    payload = {
        "model_name_or_path": args.model,
        "adapter_path": str(args.adapter) if args.adapter else "",
        "pair_count": len(pairs),
        "layers": layer_indices,
        "vectors": vectors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved {len(vectors)} steering vectors to {args.output}")


if __name__ == "__main__":
    main()

