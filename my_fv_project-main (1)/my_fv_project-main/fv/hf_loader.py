"""Unified HF model loader with quantization-aware diagnostics."""

from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
from typing import Optional

from .adapters import resolve_attn, resolve_blocks, resolve_out_proj
from .model_spec import get_model_spec


def _resolve_dtype(dtype: Optional[str], torch_module, device_str: Optional[str]) -> str:
    if dtype is None:
        return "fp32"
    if dtype not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unknown dtype: {dtype}")
    if device_str == "cpu" and dtype != "fp32":
        print(f"cpu does not support {dtype}; forcing fp32")
        return "fp32"
    return dtype


def _dtype_to_torch(dtype: str, torch_module):
    mapping = {
        "fp32": torch_module.float32,
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
    }
    return mapping[dtype]


def _resolve_quant(quant: Optional[str], model_spec: str) -> str:
    if quant is None:
        return "none"
    if quant not in {"auto", "none", "4bit", "8bit"}:
        raise ValueError(f"Unknown quant option: {quant}")
    if quant != "auto":
        return quant
    if "llama3" in model_spec.lower():
        return "4bit"
    return "none"


def _require_bitsandbytes(resolved_quant: str) -> None:
    if resolved_quant in {"4bit", "8bit"} and find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "bitsandbytes is required for 4bit/8bit quantization; "
            "install it or set --quant none"
        )


def _infer_out_proj_class(model, spec_name: str) -> Optional[str]:
    try:
        spec = get_model_spec(spec_name)
        blocks = resolve_blocks(model, spec, logger=None)
        if not blocks:
            return None
        attn = resolve_attn(blocks[0], spec, logger=None)
        out_proj = resolve_out_proj(attn, spec, logger=None)
        return out_proj.__class__.__name__
    except Exception:
        return None


def load_hf_model_and_tokenizer(
    model_name: str,
    model_spec: str,
    device: Optional[str],
    dtype: Optional[str],
    quant: str,
    device_map: Optional[str],
    trust_remote_code: bool = False,
):
    """Load HF model/tokenizer with unified quant/device handling.

    Returns (model, tokenizer, diagnostics_str).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover - runtime import check
        raise RuntimeError(f"Failed to import transformers/torch: {exc}") from exc

    resolved_dtype = _resolve_dtype(dtype, torch, device)
    resolved_quant = _resolve_quant(quant, model_spec=model_spec)
    _require_bitsandbytes(resolved_quant)

    resolved_device_map = device_map
    torch_dtype = _dtype_to_torch(resolved_dtype, torch)

    load_kwargs = {"trust_remote_code": trust_remote_code}
    if resolved_quant == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif resolved_quant == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        if resolved_dtype != "fp32":
            load_kwargs["torch_dtype"] = torch_dtype

    if resolved_device_map:
        load_kwargs["device_map"] = resolved_device_map

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    if resolved_device_map is None and device and resolved_quant not in {"4bit", "8bit"}:
        model.to(device)

    out_proj_class = _infer_out_proj_class(model, model_spec)
    has_hf_device_map = hasattr(model, "hf_device_map")

    diagnostics = {
        "model_name": model_name,
        "model_spec": model_spec,
        "resolved_quant": resolved_quant,
        "resolved_device_map": resolved_device_map,
        "resolved_dtype": resolved_dtype,
        "has_hf_device_map": has_hf_device_map,
        "out_proj_class": out_proj_class,
    }

    return model, tokenizer, diagnostics
