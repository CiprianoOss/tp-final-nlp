from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_MODEL = "fireball-nlp/fireball-qwen2.5-0.5b-lora"


def cargar_modelo():
    """Descarga y carga el modelo base junto con el adaptador LoRA."""
    print("Cargando el modelo (la primera vez puede tardar)...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
        dispositivo = torch.cuda.get_device_name(0)
    else:
        dtype = torch.float32
        device_map = {"": "cpu"}
        dispositivo = "CPU"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base_model, LORA_MODEL)
    model.eval()

    print(f"Modelo listo en {dispositivo}.")
    return model, tokenizer


def main() -> None:
    model, tokenizer = cargar_modelo()
    historial: list[dict[str, str]] = []

    print("Escribí tu mensaje. Usá 'salir' para terminar o 'limpiar' para reiniciar.")

    while True:
        try:
            prompt = input("\nVos: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not prompt:
            continue
        if prompt.lower() in {"salir", "exit", "quit"}:
            print("Hasta luego.")
            break
        if prompt.lower() == "limpiar":
            historial.clear()
            print("Conversación reiniciada.")
            continue

        historial.append({"role": "user", "content": prompt})
        texto = tokenizer.apply_chat_template(
            historial,
            tokenize=False,
            add_generation_prompt=True,
        )
        entradas = tokenizer(texto, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            salida = model.generate(
                **entradas,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        tokens_nuevos = salida[0, entradas.input_ids.shape[1] :]
        respuesta = tokenizer.decode(tokens_nuevos, skip_special_tokens=True).strip()
        print(f"\nModelo: {respuesta}")
        historial.append({"role": "assistant", "content": respuesta})


if __name__ == "__main__":
    main()
