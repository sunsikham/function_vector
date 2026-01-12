#!/usr/bin/env python3
"""Smoke test for Step6 dataset prompt construction."""

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.dataset_loader import load_pairs_antonym
from fv.slots import compute_query_predictive_slot


def compute_slot_with_fallback(prefix_str: str, full_str: str, tokenizer):
    try:
        return compute_query_predictive_slot(prefix_str, full_str, tokenizer), False
    except ValueError as exc:
        message = str(exc)
        if "Target id mismatch" not in message:
            raise
        trimmed_prefix = prefix_str.rstrip(" ")
        if trimmed_prefix == prefix_str:
            raise
        slot = compute_query_predictive_slot(trimmed_prefix, full_str, tokenizer)
        return slot, True


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Step6 dataset prompt.")
    parser.add_argument(
        "--dataset_path",
        default="datasets/processed/antonym.json",
        help="Path to antonym dataset JSON",
    )
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import transformers: {exc}")
        return 1

    pairs = load_pairs_antonym(args.dataset_path, canonical_by_input=True)
    if not pairs:
        print("No pairs loaded.")
        return 1

    rng = random.Random(args.seed)
    x_val, y_val = rng.choice(pairs)
    prefix_str = f"Q: {x_val}\nA: "
    full_str = prefix_str + y_val

    print(f"prefix_endswith_A_space: {prefix_str.endswith('A: ')}")
    if not prefix_str.endswith("A: "):
        print("FAILED: prefix_str does not end with 'A: '")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    try:
        slot_info, fallback_used = compute_slot_with_fallback(
            prefix_str, full_str, tokenizer
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    target_id = slot_info["target_id"]
    target_token = tokenizer.decode([target_id])
    print(f"target_id: {target_id}")
    print(f"target_token: {target_token!r}")
    print(f"fallback_used: {fallback_used}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
