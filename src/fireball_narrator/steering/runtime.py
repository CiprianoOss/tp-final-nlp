from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch

from fireball_narrator.modeling import get_decoder_layers


def load_steering_vectors(path: str | Path) -> dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw_vectors = payload.get("vectors", payload)
    return {int(layer): vector.float() for layer, vector in raw_vectors.items()}


@contextmanager
def apply_steering(
    model: Any,
    vectors: dict[int, torch.Tensor],
    alpha: float,
) -> Iterator[None]:
    layers = get_decoder_layers(model)
    hooks = []

    for layer_index, vector in vectors.items():
        if layer_index < 0 or layer_index >= len(layers):
            raise IndexError(
                f"Layer {layer_index} is outside the model range 0..{len(layers) - 1}"
            )

        def hook(_module: Any, _inputs: Any, output: Any, vec=vector) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            steered = hidden.clone()
            direction = vec.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
            steered[:, -1:, :] = steered[:, -1:, :] + alpha * direction
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        hooks.append(layers[layer_index].register_forward_hook(hook))

    try:
        yield
    finally:
        for handle in hooks:
            handle.remove()

