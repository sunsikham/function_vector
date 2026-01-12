#!/usr/bin/env python3
"""C1 MVP skeleton: dummy ICL prompts + slot logit extraction."""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.io import prepare_run_dirs, resolve_out_dir

DUMMY_ROWS = [
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
    def __init__(self, rows: List[Dict[str, str]]):
        self._rows = rows

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


def build_prompt(demos: List[Dict[str, str]], query: Dict[str, str]) -> Tuple[str, str]:
    demo_parts = []
    for row in demos:
        demo_parts.append(f"QUERY: {row['ex_A']}\nANSWER: {row['ex_B']}\n\n")
    demos_str = "".join(demo_parts)
    prefix_str = f"{demos_str}QUERY: {query['ex_A']}\nANSWER: "
    full_str = f"{prefix_str}{query['ex_B']}\n"
    return prefix_str, full_str


def main() -> int:
    parser = argparse.ArgumentParser(description="C1 skeleton prompt/slot/logit runner.")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)",
    )
    parser.add_argument("--n_trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--k_shot", type=int, default=2, help="Number of demos per prompt")
    parser.add_argument(
        "--csv_path",
        default="",
        help="CSV path (not used yet; dummy rows only)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier (default: auto timestamp)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/<run_id>/artifacts/c1/)",
    )
    args = parser.parse_args()

    run_info = None
    if args.out_dir:
        out_dir = resolve_out_dir(args.out_dir)
    else:
        run_info = prepare_run_dirs(args.run_id)
        out_dir = os.path.join(run_info["artifacts_dir"], "c1")
    os.makedirs(out_dir, exist_ok=True)

    if args.csv_path:
        print("CSV loading not enabled yet; using dummy rows")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import required libraries: {exc}")
        return 1

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available")
        return 1

    device = torch.device(args.device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"device: cuda ({gpu_name})")
    else:
        print("device: cpu")

    rng = random.Random(args.seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load model '{args.model}': {exc}")
        return 1

    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    provider = DummyDataProvider(DUMMY_ROWS)
    rows = provider.get_rows()
    if args.k_shot + 1 > len(rows):
        print("k_shot too large for dummy dataset")
        return 1

    results = []
    for trial_idx in range(args.n_trials):
        demos, query = provider.sample_demo_and_query(args.k_shot, rng)
        prefix_str, full_str = build_prompt(demos, query)

        prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
        full_ids = tokenizer.encode(full_str, add_special_tokens=False)

        s = len(prefix_ids)
        if s == 0 or s >= len(full_ids):
            print("Invalid prefix length for slot computation")
            return 1

        target_id = full_ids[s]
        target_token = tokenizer.convert_ids_to_tokens(target_id)
        logits_index = s - 1

        inputs = tokenizer(full_str, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        logits = outputs.logits[0]
        logit_value = logits[logits_index, target_id].item()
        prob_value = torch.softmax(logits[logits_index].float(), dim=-1)[target_id].item()

        demo_ids = [row["id"] for row in demos]
        print(
            "slot_debug: "
            f"trial={trial_idx} "
            f"s={s} "
            f"logits_index={logits_index} "
            f"target_id={target_id} "
            f"target_token={target_token}"
        )

        results.append(
            {
                "trial_idx": trial_idx,
                "demo_ids": demo_ids,
                "query_id": query["id"],
                "query_A": query["ex_A"],
                "query_B": query["ex_B"],
                "s": s,
                "target_id": target_id,
                "target_token": target_token,
                "logit_value": logit_value,
                "prob_value": prob_value,
                "prompt_preview": full_str[:120],
            }
        )

    out_path = os.path.join(out_dir, f"c1_results_n{args.n_trials}_k{args.k_shot}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)

    if run_info:
        print(f"run_id: {run_info['run_id']}")
    print(f"out_dir: {out_dir}")
    print(f"saved results: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
