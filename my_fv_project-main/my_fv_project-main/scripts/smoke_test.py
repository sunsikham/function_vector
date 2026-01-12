#!/usr/bin/env python3
"""Load a HuggingFace causal LM (default: gpt2) to validate setup."""

import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.io import prepare_run_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Load a HuggingFace causal LM.")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Model name or path (default: gpt2)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier (default: auto timestamp)",
    )
    args = parser.parse_args()

    run_info = prepare_run_dirs(args.run_id)
    log_path = os.path.join(run_info["logs_dir"], "smoke_test.log")

    # Import inside main to keep startup minimal and fail fast with a clear message.
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import transformers: {exc}")
        return 1

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load model '{args.model}': {exc}")
        return 1

    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"Device: cuda ({gpu_name})")
    else:
        print("Device: cpu")

    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits
    print(f"Logits shape: {tuple(logits.shape)}")
    assert logits.dim() == 3, "Logits must be 3D (batch, seq, vocab)."
    assert logits.shape[-1] == tokenizer.vocab_size, (
        "Logits vocab dimension must match tokenizer vocab_size."
    )

    last_token_logits = logits[0, -1]
    probs = torch.softmax(last_token_logits, dim=-1)
    top_k = 10
    top_probs, top_ids = torch.topk(probs, k=top_k)

    print(f"Top-{top_k} next-token predictions:")
    for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        print(f"id={token_id}\ttoken={token_str}\tprob={prob:.6f}")

    print(f"Loaded causal LM: {args.model}")
    log_lines = [
        f"run_id: {run_info['run_id']}",
        f"model: {args.model}",
        f"device: {device.type}",
        f"logits_shape: {tuple(logits.shape)}",
        f"top_k: {top_k}",
    ]
    for token_id, prob in zip(top_ids.tolist(), top_probs.tolist()):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        log_lines.append(f"id={token_id}\ttoken={token_str}\tprob={prob:.6f}")
    log_lines.append("status: ok")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"log saved: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
