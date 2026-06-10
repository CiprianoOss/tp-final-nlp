from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into its base model."
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-shard-size", default="4GB")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    peft_config = PeftConfig.from_pretrained(args.adapter)
    base_model_name = peft_config.base_model_name_or_path
    if not base_model_name:
        raise ValueError("The adapter does not declare base_model_name_or_path")

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.config.use_cache = True

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        args.output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    tokenizer.save_pretrained(args.output)
    print(f"Merged model saved to {args.output}")


if __name__ == "__main__":
    main()

