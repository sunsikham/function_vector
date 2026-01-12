"""Dataset loaders and samplers."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple


def load_pairs_antonym(
    dataset_path: str, canonical_by_input: bool = True
) -> List[Tuple[str, str]]:
    """Load (input, output) pairs from antonym JSON dataset."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_rows = 0
    kept_pairs = 0
    skipped_rows = 0
    dropped_due_to_dup_input = 0
    seen_inputs = set()
    pairs: List[Tuple[str, str]] = []

    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of objects")

    for row in data:
        total_rows += 1
        if not isinstance(row, dict):
            skipped_rows += 1
            continue
        x = row.get("input")
        y = row.get("output")
        if not isinstance(x, str) or not isinstance(y, str):
            skipped_rows += 1
            continue
        x = x.strip()
        y = y.strip()
        if canonical_by_input:
            if x in seen_inputs:
                dropped_due_to_dup_input += 1
                continue
            seen_inputs.add(x)
        pairs.append((x, y))
        kept_pairs += 1

    unique_inputs = len(seen_inputs) if canonical_by_input else len({p[0] for p in pairs})
    stats: Dict[str, int] = {
        "total_rows": total_rows,
        "kept_pairs": kept_pairs,
        "skipped_rows": skipped_rows,
        "unique_inputs": unique_inputs,
        "dropped_due_to_dup_input": dropped_due_to_dup_input,
    }
    print(f"load_pairs_antonym stats: {stats}")
    return pairs


def sample_demos_and_query(
    pairs: List[Tuple[str, str]], k_shot: int, seed: int
):
    """Sample k_shot demos and a distinct query from pairs."""
    if k_shot < 1:
        raise ValueError("k_shot must be >= 1")
    if len(pairs) < k_shot + 1:
        raise ValueError("Not enough pairs to sample demos and query")

    import random

    rng = random.Random(seed)
    picked = rng.sample(pairs, k_shot + 1)
    demos = picked[:k_shot]
    query = picked[-1]
    return demos, query
