"""Corruption helpers for demo permutations."""

from typing import List, Tuple


def make_corrupted_demos(
    demos: List[Tuple[str, str]], rng, ensure_derangement: bool = True
) -> List[Tuple[str, str]]:
    if len(demos) < 2:
        raise ValueError("Need at least 2 demos to corrupt")

    xs = [pair[0] for pair in demos]
    ys = [pair[1] for pair in demos]
    n = len(ys)

    if ensure_derangement:
        max_tries = 200
        for _ in range(max_tries):
            perm = rng.sample(range(n), n)
            perm_ys = [ys[i] for i in perm]
            if all(perm_ys[i] != ys[i] for i in range(n)):
                return list(zip(xs, perm_ys))

        rotated = ys[1:] + ys[:1]
        if all(rotated[i] != ys[i] for i in range(n)):
            return list(zip(xs, rotated))

    perm_ys = list(ys)
    rng.shuffle(perm_ys)
    return list(zip(xs, perm_ys))
