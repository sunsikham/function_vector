"""Prompt builders for antonym-style tasks."""

import random
from typing import List, Tuple


ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("hot", "cold"),
    ("up", "down"),
    ("big", "small"),
    ("happy", "sad"),
    ("fast", "slow"),
    ("young", "old"),
    ("light", "dark"),
    ("hard", "soft"),
    ("early", "late"),
    ("open", "close"),
    ("high", "low"),
    ("strong", "weak"),
    ("clean", "dirty"),
    ("sharp", "dull"),
]


def build_two_shot_prompt(rng: random.Random) -> Tuple[str, str, str]:
    pairs = rng.sample(ANTONYM_PAIRS, 3)
    shot_1, shot_2, query = pairs
    lines = [
        "Antonyms:",
        f"{shot_1[0]} -> {shot_1[1]}",
        f"{shot_2[0]} -> {shot_2[1]}",
        f"{query[0]} ->",
    ]
    prompt = "\n".join(lines)
    prefix_str = f"{prompt} "
    full_str = f"{prefix_str}{query[1]}"
    return prefix_str, full_str, query[1]


def build_zero_shot_prompt(rng: random.Random) -> Tuple[str, str]:
    query, answer = rng.choice(ANTONYM_PAIRS)
    lines = ["Antonyms:", f"{query} ->"]
    return "\n".join(lines), answer


def build_prompt_qa(
    demos: List[Tuple[str, str]], query: Tuple[str, str]
) -> Tuple[str, str]:
    """Build Q/A style prompt with a single trailing space after 'A:'."""
    demo_blocks = [f"Q: {x}\nA: {y}" for x, y in demos]
    query_prefix = f"Q: {query[0]}\nA: "
    if demo_blocks:
        prefix_str = "\n\n".join(demo_blocks + [query_prefix])
    else:
        prefix_str = query_prefix
    full_str = prefix_str + query[1]
    return prefix_str, full_str
