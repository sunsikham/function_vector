#!/usr/bin/env python3
"""Run a single forward pass with one hook to capture a tensor."""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.io import prepare_run_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Hook smoke test for causal LM.")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Model name or path (default: gpt2)",
    )
    parser.add_argument(
        "--block-index",
        type=int,
        default=0,
        help="GPT-2 block index to hook (default: 0)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier (default: auto timestamp)",
    )
    args = parser.parse_args()

    run_info = prepare_run_dirs(args.run_id)
    log_path = os.path.join(run_info["logs_dir"], "hook_smoke_test.log")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import required libraries: {exc}")
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load model '{args.model}': {exc}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    target_module = None
    target_name = None

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
        if 0 <= args.block_index < len(blocks):
            block = blocks[args.block_index]
            if hasattr(block, "attn"):
                attn = block.attn
                if hasattr(attn, "c_attn"):
                    target_module = attn.c_attn
                    target_name = f"transformer.h.{args.block_index}.attn.c_attn"
                elif hasattr(attn, "c_proj"):
                    target_module = attn.c_proj
                    target_name = f"transformer.h.{args.block_index}.attn.c_proj"
            if target_module is None:
                target_module = block
                target_name = f"transformer.h.{args.block_index}"

    if target_module is None:
        target_module = model
        target_name = "model"

    hook_state = {"called": False, "captured": False, "stats": None}

    def hook_fn(_module, _inputs, output):
        if hook_state["called"]:
            return
        hook_state["called"] = True
        tensor = output
        if not hasattr(tensor, "shape"):
            print("hook called, but no tensor output found")
            return

        hidden_size = getattr(model.config, "n_embd", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "hidden_size", None)
        num_heads = getattr(model.config, "n_head", None)
        if num_heads is None:
            num_heads = getattr(model.config, "num_attention_heads", None)

        if hidden_size is None or num_heads is None:
            print("hook called, but missing attention config for head split")
            return

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by n_heads"
        head_dim = hidden_size // num_heads
        assert num_heads * head_dim == hidden_size, "n_heads * head_dim must equal hidden_size"
        assert num_heads * head_dim == 768, "n_heads * head_dim must equal 768"

        if tensor.shape[-1] != 3 * hidden_size:
            print("hook called, but QKV projection shape mismatch")
            return

        _, _, value = tensor.split(hidden_size, dim=-1)
        value = value.reshape(value.shape[0], value.shape[1], num_heads, head_dim)

        stats_tensor = value.float()
        mean = stats_tensor.mean().item()
        std = stats_tensor.std(unbiased=False).item()
        norm = stats_tensor.norm().item()
        print("hook called")
        print(
            "captured: "
            f"module={target_name} "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype} "
            f"device={value.device} "
            f"n_heads={num_heads} "
            f"head_dim={head_dim} "
            f"mean={mean:.6f} "
            f"std={std:.6f} "
            f"norm={norm:.6f}"
        )
        hook_state["stats"] = {
            "module": target_name,
            "shape": tuple(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "n_heads": num_heads,
            "head_dim": head_dim,
            "mean": mean,
            "std": std,
            "norm": norm,
        }
        hook_state["captured"] = True

    handle = target_module.register_forward_hook(hook_fn)

    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        _ = model(**inputs)

    handle.remove()

    if not hook_state["called"]:
        print("훅이 안 걸림")
        return 1
    if not hook_state["captured"]:
        print("hook called, but capture failed")
        return 1

    if hook_state["stats"]:
        stats = hook_state["stats"]
        log_lines = [
            f"run_id: {run_info['run_id']}",
            f"model: {args.model}",
            f"block_index: {args.block_index}",
            f"module: {stats['module']}",
            f"shape: {stats['shape']}",
            f"dtype: {stats['dtype']}",
            f"device: {stats['device']}",
            f"n_heads: {stats['n_heads']}",
            f"head_dim: {stats['head_dim']}",
            f"mean: {stats['mean']:.6f}",
            f"std: {stats['std']:.6f}",
            f"norm: {stats['norm']:.6f}",
            "status: ok",
        ]
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"log saved: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
