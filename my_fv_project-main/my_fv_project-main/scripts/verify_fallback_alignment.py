#!/usr/bin/env python3
"""Verify slot alignment with fallback on target-id mismatch."""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.corrupt import make_corrupted_demos
from fv.dataset_loader import load_pairs_antonym, sample_demos_and_query
from fv.prompting import build_prompt_qa
from fv.slots import compute_query_predictive_slot


def compute_true_target(prefix_str: str, full_str: str, tokenizer) -> Dict[str, object]:
    prefix_ids = tokenizer(prefix_str, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_str, add_special_tokens=False).input_ids
    s = len(prefix_ids)
    if s >= len(full_ids):
        trimmed_prefix = prefix_str.rstrip(" ")
        if trimmed_prefix == prefix_str:
            raise ValueError(
                f"Invalid prefix length for true target (s={s}, seq_len={len(full_ids)})"
            )
        prefix_ids = tokenizer(trimmed_prefix, add_special_tokens=False).input_ids
        s = len(prefix_ids)
        if s >= len(full_ids):
            raise ValueError(
                f"Invalid prefix length for true target (s={s}, seq_len={len(full_ids)})"
            )
    answer = full_str[len(prefix_str) :]
    answer_with_space = answer if answer.startswith(" ") else f" {answer}"
    expected_ids = tokenizer.encode(answer_with_space, add_special_tokens=False)
    if not expected_ids:
        raise ValueError("Empty answer tokenization")
    target_id = expected_ids[0]
    prefix_target_id = full_ids[s]
    return {
        "s": s,
        "target_id": target_id,
        "full_ids": full_ids,
        "prefix_target_id": prefix_target_id,
    }


def compute_slot_with_fallback(prefix_str: str, full_str: str, tokenizer):
    mismatch = False
    fallback_used = False
    try:
        slot = compute_query_predictive_slot(prefix_str, full_str, tokenizer)
        return slot, mismatch, fallback_used
    except ValueError as exc:
        message = str(exc)
        if "Target id mismatch" not in message:
            raise
        mismatch = True
        trimmed_prefix = prefix_str.rstrip(" ")
        if trimmed_prefix == prefix_str:
            raise
        slot = compute_query_predictive_slot(trimmed_prefix, full_str, tokenizer)
        fallback_used = True
        return slot, mismatch, fallback_used


def decode_ids(tokenizer, ids: List[int]) -> List[str]:
    return [tokenizer.decode([token_id]) for token_id in ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify slot fallback alignment.")
    parser.add_argument(
        "--dataset_path",
        default="datasets/processed/antonym.json",
        help="Path to antonym dataset JSON",
    )
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--k_shot", type=int, default=2, help="K-shot demos")
    parser.add_argument("--n_checks", type=int, default=50, help="Number of checks")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--print_examples",
        type=int,
        default=5,
        help="Max failure examples to print",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import transformers: {exc}")
        return 1

    pairs = load_pairs_antonym(args.dataset_path, canonical_by_input=True)
    if len(pairs) < args.k_shot + 1:
        print(
            "Not enough pairs for sampling: "
            f"kept_pairs={len(pairs)} k_shot={args.k_shot}"
        )
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    clean_ok = 0
    clean_fail = 0
    corrupt_ok = 0
    corrupt_fail = 0
    mismatch_count_clean = 0
    mismatch_count_corrupt = 0
    fallback_used_clean = 0
    fallback_used_corrupt = 0
    prefix_spacing_fail = 0
    failure_examples: List[Dict[str, object]] = []

    for idx in range(args.n_checks):
        demos, query = sample_demos_and_query(
            pairs, args.k_shot, seed=args.seed + idx
        )
        clean_prefix, clean_full = build_prompt_qa(demos, query)
        corrupted_demos = make_corrupted_demos(
            demos, random.Random(args.seed + idx), ensure_derangement=True
        )
        corrupt_prefix, corrupt_full = build_prompt_qa(corrupted_demos, query)

        for label, prefix_str, full_str in (
            ("clean", clean_prefix, clean_full),
            ("corrupted", corrupt_prefix, corrupt_full),
        ):
            if not prefix_str.endswith("A: "):
                prefix_spacing_fail += 1

            try:
                true_info = compute_true_target(prefix_str, full_str, tokenizer)
            except ValueError as exc:
                failure_examples.append(
                    {
                        "label": label,
                        "seed": args.seed + idx,
                        "query": query,
                        "prefix_tail": repr(prefix_str[-40:]),
                        "error": str(exc),
                    }
                )
                if label == "clean":
                    clean_fail += 1
                else:
                    corrupt_fail += 1
                continue

            try:
                slot_info, mismatch, fallback = compute_slot_with_fallback(
                    prefix_str, full_str, tokenizer
                )
            except ValueError as exc:
                failure_examples.append(
                    {
                        "label": label,
                        "seed": args.seed + idx,
                        "query": query,
                        "prefix_tail": repr(prefix_str[-40:]),
                        "error": str(exc),
                    }
                )
                if label == "clean":
                    clean_fail += 1
                else:
                    corrupt_fail += 1
                continue

            if label == "clean":
                if mismatch:
                    mismatch_count_clean += 1
                if fallback:
                    fallback_used_clean += 1
            else:
                if mismatch:
                    mismatch_count_corrupt += 1
                if fallback:
                    fallback_used_corrupt += 1

            true_target_id = true_info["target_id"]
            slot_target_id = slot_info["target_id"]
            match = true_target_id == slot_target_id

            if match:
                if label == "clean":
                    clean_ok += 1
                else:
                    corrupt_ok += 1
            else:
                if label == "clean":
                    clean_fail += 1
                else:
                    corrupt_fail += 1

                s = true_info["s"]
                full_ids = true_info["full_ids"]
                prefix_target_id = true_info["prefix_target_id"]
                start = max(s - 3, 0)
                end = min(s + 4, len(full_ids))
                context_ids = full_ids[start:end]
                failure_examples.append(
                    {
                        "label": label,
                        "seed": args.seed + idx,
                        "query": query,
                        "prefix_tail": repr(prefix_str[-40:]),
                        "true_target_id": true_target_id,
                        "true_target_token": repr(tokenizer.decode([true_target_id])),
                        "prefix_target_id": prefix_target_id,
                        "prefix_target_token": repr(
                            tokenizer.decode([prefix_target_id])
                        ),
                        "slot_target_id": slot_target_id,
                        "slot_target_token": repr(tokenizer.decode([slot_target_id])),
                        "context_ids": context_ids,
                        "context_tokens": decode_ids(tokenizer, context_ids),
                    }
                )

    total_checks = args.n_checks
    print(f"total_checks: {total_checks}")
    print(f"clean_ok: {clean_ok}")
    print(f"clean_fail: {clean_fail}")
    print(f"corrupted_ok: {corrupt_ok}")
    print(f"corrupted_fail: {corrupt_fail}")
    print(f"mismatch_count_clean: {mismatch_count_clean}")
    print(f"mismatch_count_corrupt: {mismatch_count_corrupt}")
    print(f"fallback_used_clean: {fallback_used_clean}")
    print(f"fallback_used_corrupt: {fallback_used_corrupt}")
    print(f"prefix_spacing_fail: {prefix_spacing_fail}")
    print(f"prefix_endswith_A_space: {prefix_spacing_fail == 0}")

    if clean_fail or corrupt_fail:
        print("FAILURES:")
        for example in failure_examples[: args.print_examples]:
            print(example)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
