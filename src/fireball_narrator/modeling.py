from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model_and_tokenizer(
    model_name_or_path: str,
    adapter_path: str | Path | None = None,
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    if load_in_4bit and not torch.cuda.is_available():
        raise RuntimeError("4-bit loading requires a CUDA GPU")

    quantization_config = None
    if load_in_4bit:
        compute_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map: str | dict[str, str] = "auto"
    else:
        dtype = torch.float32
        device_map = {"": "cpu"}

    tokenizer_source = str(adapter_path or model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def get_decoder_layers(model: Any) -> Any:
    candidate_paths = (
        "model.layers",
        "model.model.layers",
        "base_model.model.model.layers",
        "base_model.model.layers",
    )
    for path in candidate_paths:
        current = model
        try:
            for attribute in path.split("."):
                current = getattr(current, attribute)
        except AttributeError:
            continue
        if current is not None and len(current) > 0:
            return current
    raise AttributeError("Could not locate transformer decoder layers")


def model_input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device

