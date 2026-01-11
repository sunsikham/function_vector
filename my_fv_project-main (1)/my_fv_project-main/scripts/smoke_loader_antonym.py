#!/usr/bin/env python3
"""Smoke test for antonym dataset loader."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.dataset_loader import load_pairs_antonym, sample_demos_and_query


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test antonym loader.")
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
    print(f"demos (k={args.k_shot}): {demos}")
    print(f"query: {query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
