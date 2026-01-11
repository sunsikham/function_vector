"""Model specifications for adapter resolution."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf_model_id: str
    prepend_bos: bool
    blocks_path: str
    attn_path_in_block: str
    out_proj_path_in_attn: str
    n_layers: Optional[int]
    n_heads: Optional[int]
    head_dim: Optional[int]
    hidden_size: Optional[int]
    dtype: Optional[str]
    device_map: Optional[str]
    quantization: Optional[Dict[str, object]]


MODEL_SPECS: Dict[str, ModelSpec] = {
    "gpt2": ModelSpec(
        name="gpt2",
        hf_model_id="gpt2",
        prepend_bos=False,
        blocks_path="transformer.h",
        attn_path_in_block="attn",
        out_proj_path_in_attn="c_proj",
        n_layers=12,
        n_heads=12,
        head_dim=64,
        hidden_size=768,
        dtype="fp32",
        device_map=None,
        quantization=None,
    ),
    "llama3": ModelSpec(
        name="llama3",
        hf_model_id="llama3",
        prepend_bos=True,
        blocks_path="model.layers",
        attn_path_in_block="self_attn",
        out_proj_path_in_attn="o_proj",
        n_layers=None,
        n_heads=None,
        head_dim=None,
        hidden_size=None,
        dtype=None,
        device_map=None,
        quantization=None,
    ),
    "llama3_wrapped": ModelSpec(
        name="llama3_wrapped",
        hf_model_id="llama3_wrapped",
        prepend_bos=True,
        blocks_path="model.model.layers",
        attn_path_in_block="self_attn",
        out_proj_path_in_attn="o_proj",
        n_layers=None,
        n_heads=None,
        head_dim=None,
        hidden_size=None,
        dtype=None,
        device_map=None,
        quantization=None,
    ),
}


def get_model_spec(name_or_id: str) -> ModelSpec:
    key = name_or_id.strip()
    if key in MODEL_SPECS:
        return MODEL_SPECS[key]
    for spec in MODEL_SPECS.values():
        if key == spec.hf_model_id:
            return spec
    raise ValueError(f"Unsupported model spec: '{name_or_id}'")
