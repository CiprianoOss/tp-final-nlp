from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from fireball_narrator.config import load_yaml


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def common_prefix_length(left: list[int], right: list[int]) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def tokenize_example(
    example: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int] | bool]:
    messages = example["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        return {"input_ids": [], "attention_mask": [], "labels": [], "valid": False}

    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    prefix_length = common_prefix_length(prompt_ids, full_ids)
    labels = [-100] * prefix_length + full_ids[prefix_length:]

    if len(full_ids) > max_seq_length:
        start = len(full_ids) - max_seq_length
        full_ids = full_ids[start:]
        labels = labels[start:]

    valid = bool(full_ids) and any(label != -100 for label in labels)
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "valid": valid,
    }


class CausalLMCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []

        for feature in features:
            padding = max_length - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def prepare_dataset(
    dataset: Dataset,
    tokenizer: Any,
    max_seq_length: int,
) -> Dataset:
    columns = dataset.column_names
    tokenized = dataset.map(
        lambda row: tokenize_example(row, tokenizer, max_seq_length),
        remove_columns=columns,
        desc="Tokenizing chats",
    )
    tokenized = tokenized.filter(lambda row: row["valid"], desc="Removing empty labels")
    return tokenized.remove_columns(["valid"])


def make_quantization_config(config: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not config.get("load_in_4bit", True):
        return None
    dtype = parse_dtype(config.get("bnb_4bit_compute_dtype", "bfloat16"))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_compute_dtype=dtype,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen on FIREBALL with QLoRA.")
    parser.add_argument(
        "--config",
        default="configs/qwen3_4b_qlora.yaml",
        help="YAML training configuration.",
    )
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_cfg = config["model"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
    quant_cfg = config.get("quantization", {})

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires an NVIDIA CUDA GPU")

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": data_cfg["train_file"]}
    validation_path = data_cfg.get("validation_file")
    if validation_path and Path(validation_path).exists():
        data_files["validation"] = validation_path
    raw = load_dataset("json", data_files=data_files)
    max_seq_length = int(data_cfg.get("max_seq_length", 2048))
    train_dataset = prepare_dataset(raw["train"], tokenizer, max_seq_length)
    eval_dataset = (
        prepare_dataset(raw["validation"], tokenizer, max_seq_length)
        if "validation" in raw
        else None
    )

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    quantization_config = make_quantization_config(quant_cfg)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        quantization_config=quantization_config,
        torch_dtype=(
            parse_dtype(quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
            if quantization_config
            else "auto"
        ),
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
    )
    if quantization_config:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_cfg.get(
                "gradient_checkpointing",
                True,
            ),
        )

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 32)),
        lora_alpha=int(lora_cfg.get("alpha", 64)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg["target_modules"]),
    )
    model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    bf16 = torch.cuda.is_bf16_supported()
    evaluation_strategy = "steps" if eval_dataset is not None else "no"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 2)),
        per_device_train_batch_size=int(
            train_cfg.get("per_device_train_batch_size", 1)
        ),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(
            train_cfg.get("gradient_accumulation_steps", 16)
        ),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy=evaluation_strategy,
        eval_steps=int(train_cfg.get("eval_steps", 200)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=not bf16,
        seed=seed,
        data_seed=seed,
        report_to=train_cfg.get("report_to", "none"),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id),
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    metrics = dict(result.metrics)
    metrics["train_examples"] = len(train_dataset)
    metrics["eval_examples"] = len(eval_dataset) if eval_dataset is not None else 0
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


if __name__ == "__main__":
    main()

