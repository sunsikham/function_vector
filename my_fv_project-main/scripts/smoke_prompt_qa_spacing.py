#!/usr/bin/env python3
"""Smoke test for Q/A prompt spacing."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.dataset_loader import load_pairs_antonym, sample_demos_and_query
from fv.prompting import build_prompt_qa


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Q/A prompt spacing.")
    parser.add_argument(
        "--dataset_path",
        default="datasets/processed/antonym.json",
        help="Path to antonym dataset JSON",
    )
    parser.add_argument("--k_shot", type=int, default=2, help="K-shot demos")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    pairs = load_pairs_antonym(args.dataset_path, canonical_by_input=True)
    if len(pairs) < args.k_shot + 1:
        print(
            "Not enough pairs for sampling: "
            f"kept_pairs={len(pairs)} k_shot={args.k_shot}"
        )
        return 1

    demos, query = sample_demos_and_query(pairs, args.k_shot, args.seed)
    prefix_str, full_str = build_prompt_qa(demos, query)

    print(f"prefix_suffix: {prefix_str[-3:]}")
    print(f"prefix_len: {len(prefix_str)}")
    print(f"full_len: {len(full_str)}")

    if not prefix_str.endswith("A: "):
        print("FAILED: prefix_str does not end with 'A: '")
        return 1
    if prefix_str.endswith("A:  "):
        print("FAILED: prefix_str ends with two spaces after 'A:'")
        return 1
    if not full_str.startswith(prefix_str):
        print("FAILED: full_str does not start with prefix_str")
        return 1
    if prefix_str[-3:] != "A: ":
        print(f"FAILED: prefix_str[-3:]='{prefix_str[-3:]}'")
        return 1

    print("OK: spacing checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
