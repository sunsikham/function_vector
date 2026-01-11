#!/usr/bin/env python3
"""Smoke test for spec-based out_proj resolution and pre-hook invocation."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.adapters import infer_head_dims, resolve_attn, resolve_blocks, resolve_out_proj
from fv.model_spec import get_model_spec


def parse_dtype(dtype_str: str, torch_module, device_str: str):
    mapping = {
        "fp32": torch_module.float32,
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    dtype = mapping[dtype_str]
    if device_str == "cpu" and dtype != torch_module.float32:
        print(f"cpu does not support {dtype_str}; forcing fp32")
        dtype = torch_module.float32
    return dtype


def resolve_device(device_str: str, torch_module):
    if device_str == "auto":
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if getattr(torch_module.backends, "mps", None) and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    if device_str == "cuda":
        if not torch_module.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch_module.device("cuda")
    if device_str == "mps":
        if not getattr(torch_module.backends, "mps", None) or not torch_module.backends.mps.is_available():
            raise ValueError("MPS requested but not available")
        return torch_module.device("mps")
    if device_str == "cpu":
        return torch_module.device("cpu")
    raise ValueError(f"Unknown device: {device_str}")


def resolve_hidden_size(model, spec_name: str):
    try:
        dims = infer_head_dims(model, spec_name=spec_name)
        return dims["hidden_size"]
    except Exception as exc:
        print(f"infer_head_dims failed; falling back to model.config.hidden_size: {exc}")
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Unable to infer hidden_size from model config")
    return hidden_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke: resolve out_proj via model spec.")
    parser.add_argument("--model", default="gpt2", help="Model name or path")
    parser.add_argument("--model_spec", default="gpt2", help="Model spec name")
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Dtype (default: fp32)",
    )
    parser.add_argument("--device_map", default=None, help="HuggingFace device_map")
    parser.add_argument("--prompt", default="Hello", help="Prompt string")
    parser.add_argument(
        "--negative",
        action="store_true",
        help="Force bad blocks_path resolution and expect failure",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import required libraries: {exc}")
        return 1

    try:
        device = resolve_device(args.device, torch)
    except ValueError as exc:
        print(str(exc))
        return 1

    try:
        dtype = parse_dtype(args.dtype, torch, device.type)
    except ValueError as exc:
        print(str(exc))
        return 1

    try:
        spec = get_model_spec(args.model_spec)
    except ValueError as exc:
        print(str(exc))
        return 1

    load_kwargs = {"torch_dtype": dtype}
    if args.device_map:
        load_kwargs["device_map"] = args.device_map

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load model '{args.model}': {exc}")
        return 1

    if args.device_map is None:
        model.to(device)

    model.eval()

    if args.negative:
        bad_spec = replace(spec, blocks_path="model.NOT_A_REAL_PATH")
        try:
            resolve_blocks(model, bad_spec, logger=print)
        except ValueError as exc:
            print(f"negative mode: expected failure: {exc}")
            return 0
        print("negative mode: resolve unexpectedly succeeded")
        return 1

    try:
        blocks = resolve_blocks(model, spec, logger=print)
    except ValueError as exc:
        print(str(exc))
        if args.model_spec == "llama3":
            example = (
                "python scripts/smoke_resolve_spec_outproj.py "
                f"--model {args.model} --model_spec llama3_wrapped "
                f"--layer {args.layer} --device {args.device} --dtype {args.dtype}"
            )
            print(f"Try llama3_wrapped: {example}")
        return 1

    if args.layer < 0 or args.layer >= len(blocks):
        print(f"layer out of range: {args.layer} not in [0, {len(blocks) - 1}]")
        return 1

    try:
        attn = resolve_attn(blocks[args.layer], spec, logger=print)
        out_proj = resolve_out_proj(attn, spec, logger=print)
    except ValueError as exc:
        print(str(exc))
        return 1

    try:
        hidden_size = resolve_hidden_size(model, args.model_spec)
    except ValueError as exc:
        print(str(exc))
        return 1

    hook_state = {"called": False, "dim_ok": True, "error": None}

    def pre_hook(_module, inputs):
        hook_state["called"] = True
        if not inputs:
            hook_state["dim_ok"] = False
            hook_state["error"] = "pre-hook received empty inputs"
            return
        tensor = inputs[0]
        if not hasattr(tensor, "shape"):
            hook_state["dim_ok"] = False
            hook_state["error"] = "pre-hook input is not a tensor"
            return
        last_dim = tensor.shape[-1] if tensor.dim() > 0 else None
        if last_dim != hidden_size:
            hook_state["dim_ok"] = False
            hook_state["error"] = (
                f"input last dim mismatch: got={last_dim} expected={hidden_size}"
            )
        with torch.no_grad():
            mean = tensor.float().mean().item()
            std = tensor.float().std().item()
            norm = tensor.float().norm().item()
        print(
            "out_proj pre-hook input: "
            f"shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype} "
            f"device={tensor.device} "
            f"mean={mean:.6f} "
            f"std={std:.6f} "
            f"norm={norm:.6f}"
        )

    handle = out_proj.register_forward_pre_hook(pre_hook)
    try:
        inputs = tokenizer(args.prompt, return_tensors="pt")
        if args.device_map is None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            _ = model(**inputs)
    finally:
        handle.remove()

    if not hook_state["called"]:
        print("out_proj pre-hook was not called")
        return 1
    if not hook_state["dim_ok"]:
        print(f"out_proj pre-hook validation failed: {hook_state['error']}")
        return 1

    print("resolve success: out_proj pre-hook executed and input dims validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
