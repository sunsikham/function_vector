"""Data provider interfaces and dummy rows."""

from typing import Dict, List, Tuple
import random


DUMMY_ROWS: List[Dict[str, str]] = [
    {"id": "r01", "ex_A": "red", "ex_B": "color"},
    {"id": "r02", "ex_A": "dog", "ex_B": "animal"},
    {"id": "r03", "ex_A": "car", "ex_B": "vehicle"},
    {"id": "r04", "ex_A": "apple", "ex_B": "fruit"},
    {"id": "r05", "ex_A": "table", "ex_B": "furniture"},
    {"id": "r06", "ex_A": "Paris", "ex_B": "city"},
    {"id": "r07", "ex_A": "violin", "ex_B": "instrument"},
    {"id": "r08", "ex_A": "rose", "ex_B": "flower"},
    {"id": "r09", "ex_A": "oak", "ex_B": "tree"},
    {"id": "r10", "ex_A": "salmon", "ex_B": "fish"},
    {"id": "r11", "ex_A": "blue", "ex_B": "color"},
    {"id": "r12", "ex_A": "cat", "ex_B": "animal"},
    {"id": "r13", "ex_A": "bus", "ex_B": "vehicle"},
    {"id": "r14", "ex_A": "banana", "ex_B": "fruit"},
    {"id": "r15", "ex_A": "chair", "ex_B": "furniture"},
    {"id": "r16", "ex_A": "Tokyo", "ex_B": "city"},
    {"id": "r17", "ex_A": "piano", "ex_B": "instrument"},
    {"id": "r18", "ex_A": "tulip", "ex_B": "flower"},
    {"id": "r19", "ex_A": "maple", "ex_B": "tree"},
    {"id": "r20", "ex_A": "tuna", "ex_B": "fish"},
]


class DummyDataProvider:
    def __init__(self, rows: List[Dict[str, str]] = None):
        self._rows = list(rows) if rows is not None else list(DUMMY_ROWS)

    def get_rows(self) -> List[Dict[str, str]]:
        return list(self._rows)

    def sample_demo_and_query(
        self, k_shot: int, rng: random.Random
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        if k_shot < 0:
            raise ValueError("k_shot must be >= 0")
        if k_shot + 1 > len(self._rows):
            raise ValueError("Not enough rows for k_shot demos + 1 query")
        sampled = rng.sample(self._rows, k_shot + 1)
        demos = sampled[:k_shot]
        query = sampled[-1]
        return demos, query
