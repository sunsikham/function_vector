#!/usr/bin/env python3
"""STEP B: Build corrupted demos and compare baseline logits."""

import argparse
import os
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.corrupt import make_corrupted_demos
from fv.io import prepare_run_dirs, resolve_out_dir, save_json
from fv.prompting import ANTONYM_PAIRS
from fv.slots import compute_query_predictive_slot


def make_logger(log_path: str):
    log_file = open(log_path, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    return log, log_file


def build_prompt(demos, query_pair):
    lines = ["Antonyms:"]
    for x_val, y_val in demos:
        lines.append(f"{x_val} -> {y_val}")
    lines.append(f"{query_pair[0]} ->")
    prompt = "\n".join(lines)
    prefix_str = f"{prompt} "
    full_str = f"{prefix_str}{query_pair[1]}"
    return prefix_str, full_str


def summarize(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP B corrupted demo baseline.")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials (default: 50)")
    parser.add_argument(
        "--n_icl_examples",
        type=int,
        default=2,
        help="Number of ICL demos per prompt (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier (default: auto timestamp)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/<run_id>/artifacts/)",
    )
    parser.add_argument(
        "--n_save_samples",
        type=int,
        default=5,
        help="Number of samples to save with full text (default: 5)",
    )
    args = parser.parse_args()

    if args.n_icl_examples < 1:
        print("n_icl_examples must be >= 1")
        return 1
    if args.n_trials < 1:
        print("n_trials must be >= 1")
        return 1

    run_info = prepare_run_dirs(args.run_id)
    if args.out_dir:
        artifacts_dir = resolve_out_dir(args.out_dir)
    else:
        artifacts_dir = run_info["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(run_info["logs_dir"], "stepB_corrupted_baseline.log")
    log, log_file = make_logger(log_path)

    log("stepB corrupted baseline start")
    log(f"run_id: {run_info['run_id']}")
    log(f"artifacts_dir: {artifacts_dir}")
    log(f"log_path: {log_path}")
    log(f"model: {args.model}")
    log(f"n_trials: {args.n_trials}")
    log(f"n_icl_examples: {args.n_icl_examples}")
    log(f"seed: {args.seed}")
    log(f"n_save_samples: {args.n_save_samples}")

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        log(f"Failed to import required libraries: {exc}")
        log_file.close()
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except Exception as exc:  # pragma: no cover - runtime load check
        log(f"Failed to load model '{args.model}': {exc}")
        log_file.close()
        return 1

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.n_icl_examples + 1 > len(ANTONYM_PAIRS):
        log("Not enough antonym pairs for requested demos + query.")
        log_file.close()
        return 1

    rng = random.Random(args.seed)
    results = []
    samples = []
    clean_logits = []
    corrupted_logits = []
    clean_probs = []
    corrupted_probs = []
    delta_logits = []
    delta_probs = []

    save_limit = min(args.n_save_samples, args.n_trials)

    for trial_idx in range(args.n_trials):
        pairs = rng.sample(ANTONYM_PAIRS, args.n_icl_examples + 1)
        demos = pairs[: args.n_icl_examples]
        query = pairs[-1]

        clean_prefix_str, clean_full_str = build_prompt(demos, query)
        try:
            corrupted_demos = make_corrupted_demos(demos, rng, ensure_derangement=True)
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1
        corrupted_prefix_str, corrupted_full_str = build_prompt(corrupted_demos, query)

        try:
            clean_slot = compute_query_predictive_slot(
                clean_prefix_str, clean_full_str, tokenizer
            )
            corrupted_slot = compute_query_predictive_slot(
                corrupted_prefix_str, corrupted_full_str, tokenizer
            )
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1

        clean_target_id = clean_slot["target_id"]
        corrupted_target_id = corrupted_slot["target_id"]
        if clean_target_id != corrupted_target_id:
            log(
                "target_id mismatch between clean/corrupted: "
                f"clean={clean_target_id} corrupted={corrupted_target_id}"
            )
            log_file.close()
            return 1

        target_token = clean_slot["target_token"]
        answer_str = query[1]

        def score_prefix(prefix_str: str):
            inputs = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                last_indices = torch.tensor(
                    [inputs["input_ids"].shape[1] - 1], device=device
                )
            else:
                last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(inputs["input_ids"].shape[0], device=device)

            with torch.inference_mode():
                outputs = model(**inputs)
            logits = outputs.logits
            last_logits = logits[batch_indices, last_indices]
            logit_val = last_logits[0, clean_target_id].item()
            prob_val = torch.softmax(last_logits[0].float(), dim=-1)[
                clean_target_id
            ].item()
            return logit_val, prob_val

        clean_logit, clean_prob = score_prefix(clean_prefix_str)
        corrupted_logit, corrupted_prob = score_prefix(corrupted_prefix_str)
        delta_logit = clean_logit - corrupted_logit
        delta_prob = clean_prob - corrupted_prob

        clean_logits.append(clean_logit)
        corrupted_logits.append(corrupted_logit)
        clean_probs.append(clean_prob)
        corrupted_probs.append(corrupted_prob)
        delta_logits.append(delta_logit)
        delta_probs.append(delta_prob)

        demo_map_clean = [{"x": x_val, "y": y_val} for x_val, y_val in demos]
        demo_map_corrupted = [{"x": x_val, "y": y_val} for x_val, y_val in corrupted_demos]

        results.append(
            {
                "trial_idx": trial_idx,
                "clean_prefix_str": summarize(clean_prefix_str),
                "corrupted_prefix_str": summarize(corrupted_prefix_str),
                "answer_str": answer_str,
                "target_id": clean_target_id,
                "target_token": target_token,
                "clean_logit": clean_logit,
                "corrupted_logit": corrupted_logit,
                "clean_prob": clean_prob,
                "corrupted_prob": corrupted_prob,
                "delta_logit": delta_logit,
                "delta_prob": delta_prob,
                "demo_map_clean": demo_map_clean,
                "demo_map_corrupted": demo_map_corrupted,
            }
        )

        if trial_idx < save_limit:
            samples.append(
                {
                    "trial_idx": trial_idx,
                    "clean_prefix_str": clean_prefix_str,
                    "clean_full_str": clean_full_str,
                    "corrupted_prefix_str": corrupted_prefix_str,
                    "corrupted_full_str": corrupted_full_str,
                    "answer_str": answer_str,
                    "target_id": clean_target_id,
                    "target_token": target_token,
                    "demo_map_clean": demo_map_clean,
                    "demo_map_corrupted": demo_map_corrupted,
                }
            )

    summary = {
        "n_trials": args.n_trials,
        "mean_clean_logit": mean(clean_logits),
        "mean_corrupted_logit": mean(corrupted_logits),
        "mean_delta_logit": mean(delta_logits),
        "std_clean_logit": std(clean_logits),
        "std_corrupted_logit": std(corrupted_logits),
        "std_delta_logit": std(delta_logits),
        "mean_clean_prob": mean(clean_probs),
        "mean_corrupted_prob": mean(corrupted_probs),
        "mean_delta_prob": mean(delta_probs),
        "direction_ok": mean(delta_logits) > 0.0,
        "seed": args.seed,
        "n_icl_examples": args.n_icl_examples,
    }

    samples_path = os.path.join(artifacts_dir, "corrupted_samples.json")
    results_path = os.path.join(artifacts_dir, "baseline_clean_vs_corrupted.json")

    save_json(samples_path, {"samples": samples})
    save_json(results_path, {"summary": summary, "trials": results})

    log(
        "summary: "
        f"mean_clean_logit={summary['mean_clean_logit']:.6f} "
        f"mean_corrupted_logit={summary['mean_corrupted_logit']:.6f} "
        f"mean_delta_logit={summary['mean_delta_logit']:.6f}"
    )
    log(
        "summary_prob: "
        f"mean_clean_prob={summary['mean_clean_prob']:.6f} "
        f"mean_corrupted_prob={summary['mean_corrupted_prob']:.6f} "
        f"mean_delta_prob={summary['mean_delta_prob']:.6f}"
    )
    log(f"direction_ok: {summary['direction_ok']}")
    log(f"saved samples: {samples_path}")
    log(f"saved results: {results_path}")
    log_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
