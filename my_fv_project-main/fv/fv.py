"""FV synthesis utilities."""

from typing import List, Tuple


def parse_heads(heads_str: str) -> List[Tuple[int, int]]:
    heads = []
    for part in heads_str.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid head entry: '{part}'")
        layer_str, head_str = part.split(":", 1)
        heads.append((int(layer_str), int(head_str)))
    return heads


def build_fv(mean_activations, selected_heads: List[int], head_dim: int, resid_dim: int):
    fv = mean_activations.new_zeros((resid_dim,))
    for head in selected_heads:
        start = head * head_dim
        end = start + head_dim
        fv[start:end] = mean_activations[head]
    return fv
