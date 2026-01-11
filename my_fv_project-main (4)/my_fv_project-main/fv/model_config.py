#!/usr/bin/env python3
"""Model-specific config lookup for hooks and shapes."""

from typing import Dict, Optional

from .model_spec import get_model_spec


def _normalize_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def _spec_to_config(spec) -> Dict[str, object]:
    return {
        "n_heads": spec.n_heads,
        "head_dim": spec.head_dim,
        "resid_dim": spec.hidden_size,
        "hook_target": (
            f"{spec.blocks_path}.{{layer}}."
            f"{spec.attn_path_in_block}.{spec.out_proj_path_in_attn}"
        ),
        "hook_type": "pre",
        "reshape": "resid_to_heads",
    }


def get_model_config(model_name: str) -> Optional[Dict[str, object]]:
    normalized = _normalize_name(model_name)
    try:
        spec = get_model_spec(normalized)
    except ValueError:
        return None
    return _spec_to_config(spec)
