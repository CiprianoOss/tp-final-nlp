from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fireball_narrator.data.formatting import SYSTEM_PROMPT
from fireball_narrator.decoding import build_bad_words_ids, read_forbidden_terms
from fireball_narrator.modeling import load_model_and_tokenizer, model_input_device
from fireball_narrator.steering.runtime import apply_steering, load_steering_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with activation steering.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--steering-vector", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--bad-words-file",
        type=Path,
        help="Optional newline-separated terms blocked during decoding.",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(
        args.model,
        adapter_path=args.adapter,
        load_in_4bit=args.load_in_4bit,
    )
    vectors = load_steering_vectors(args.steering_vector)
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
    bad_words_ids = (
        build_bad_words_ids(tokenizer, read_forbidden_terms(args.bad_words_file))
        if args.bad_words_file
        else None
    )

    with torch.inference_mode(), apply_steering(model, vectors, args.alpha):
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=bad_words_ids,
        )
    generated = output[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
