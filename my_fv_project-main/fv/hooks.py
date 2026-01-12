"""Hook utilities for capturing standardized activations."""

from typing import Tuple

from .adapters import resolve_attn, resolve_blocks, resolve_out_proj
from .model_spec import get_model_spec


def get_c_proj_pre_hook(model, layer: int) -> Tuple[object, str]:
    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("Model does not expose transformer.h blocks")
    if layer < 0 or layer >= len(model.transformer.h):
        raise ValueError("Layer index out of range")
    block = model.transformer.h[layer]
    if not hasattr(block, "attn") or not hasattr(block.attn, "c_proj"):
        raise ValueError("Could not find transformer.h.{layer}.attn.c_proj".format(layer=layer))
    return block.attn.c_proj, f"transformer.h.{layer}.attn.c_proj"


def get_out_proj_pre_hook_target(
    model, layer_idx: int, spec_name: str = "gpt2", logger=None
) -> Tuple[object, str]:
    """Find attention output projection module for pre-hook attachment."""
    spec = get_model_spec(spec_name)
    blocks = resolve_blocks(model, spec, logger=logger)
    if layer_idx < 0 or layer_idx >= len(blocks):
        raise ValueError("Layer index out of range")
    block = blocks[layer_idx]
    attn_module = resolve_attn(block, spec, logger=logger)
    out_proj = resolve_out_proj(attn_module, spec, logger=logger)
    path = (
        f"{spec.blocks_path}.{layer_idx}."
        f"{spec.attn_path_in_block}.{spec.out_proj_path_in_attn}"
    )
    return out_proj, path


def reshape_resid_to_heads(tensor, n_heads: int, head_dim: int, resid_dim: int):
    if tensor is None or not hasattr(tensor, "shape"):
        raise ValueError("Hook input is not a tensor")
    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D input, got {tuple(tensor.shape)}")
    if tensor.shape[-1] != resid_dim:
        raise ValueError(f"Expected last dim {resid_dim}, got {tensor.shape[-1]}")
    reshaped = tensor.reshape(tensor.shape[0], tensor.shape[1], n_heads, head_dim)
    return reshaped
