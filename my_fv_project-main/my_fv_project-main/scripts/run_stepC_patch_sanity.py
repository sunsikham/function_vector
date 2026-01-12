#!/usr/bin/env python3
"""STEP C: Patch sanity checks (self-replace no-op)."""

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
from fv.hooks import get_c_proj_pre_hook, reshape_resid_to_heads
from fv.io import prepare_run_dirs, resolve_out_dir, save_json
from fv.model_config import get_model_config
from fv.patch import make_cproj_head_replacer
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


def compute_mean_head_vec(
    prefixes,
    model,
    tokenizer,
    device,
    n_heads,
    head_dim,
    resid_dim,
    head_idx,
    token_idx,
    target_module,
):
    import torch

    sum_vec = torch.zeros((head_dim,), device=device)
    count = 0
    state = {"current": None, "errors": []}

    def pre_hook(_module, inputs):
        tensor = inputs[0] if inputs else None
        try:
            reshaped = reshape_resid_to_heads(tensor, n_heads, head_dim, resid_dim)
        except ValueError as exc:
            state["errors"].append(str(exc))
            return
        seq_len = reshaped.shape[1]
        t_idx = token_idx if token_idx >= 0 else seq_len + token_idx
        if t_idx < 0 or t_idx >= seq_len:
            state["errors"].append("token_idx out of range")
            return
        head_vec = reshaped[:, t_idx, head_idx, :]
        state["current"] = head_vec.detach().mean(dim=0)

    handle = target_module.register_forward_pre_hook(pre_hook)
    for prefix in prefixes:
        state["current"] = None
        inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            _ = model(**inputs)
        if state["errors"]:
            handle.remove()
            raise ValueError("; ".join(state["errors"]))
        if state["current"] is None:
            handle.remove()
            raise ValueError("Failed to capture head vector")
        sum_vec += state["current"]
        count += 1
    handle.remove()

    if count == 0:
        raise ValueError("No prefixes provided for mean computation")
    return sum_vec / count


def mean_abs(values):
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP C patch sanity check.")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default: 0)")
    parser.add_argument("--head", type=int, default=0, help="Head index (default: 0)")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of trials (default: 5)")
    parser.add_argument(
        "--mode",
        default="self",
        choices=["self", "clean_mean", "corrupted_mean"],
        help="Patch mode (default: self)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Optional scale for replace_vec (default: None)",
    )
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    args = parser.parse_args()

    if args.n_trials < 1:
        print("n_trials must be >= 1")
        return 1

    run_info = prepare_run_dirs(args.run_id)
    if args.out_dir:
        artifacts_dir = resolve_out_dir(args.out_dir)
    else:
        artifacts_dir = run_info["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(run_info["logs_dir"], "stepC_patch_sanity.log")
    log, log_file = make_logger(log_path)

    log("stepC patch sanity start")
    log(f"run_id: {run_info['run_id']}")
    log(f"artifacts_dir: {artifacts_dir}")
    log(f"log_path: {log_path}")
    log(f"model: {args.model}")
    log(f"layer: {args.layer}")
    log(f"head: {args.head}")
    log(f"n_trials: {args.n_trials}")
    log(f"mode: {args.mode}")
    if args.alpha is not None:
        log(f"alpha: {args.alpha}")

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        log(f"Failed to import required libraries: {exc}")
        log_file.close()
        return 1

    model_cfg = get_model_config(args.model)
    if model_cfg is None:
        log(f"No model config found for '{args.model}'")
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

    try:
        target_module, target_name = get_c_proj_pre_hook(model, args.layer)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    n_heads = int(model_cfg["n_heads"])
    head_dim = int(model_cfg["head_dim"])
    resid_dim = int(model_cfg["resid_dim"])

    if args.head < 0 or args.head >= n_heads:
        log("head index out of range")
        log_file.close()
        return 1

    if args.n_trials + 1 > len(ANTONYM_PAIRS):
        log("Not enough antonym pairs for requested trials.")
        log_file.close()
        return 1

    rng = random.Random(args.seed)
    trials = []

    for trial_idx in range(args.n_trials):
        pairs = rng.sample(ANTONYM_PAIRS, 3)
        demos = pairs[:2]
        query = pairs[-1]

        clean_prefix_str, clean_full_str = build_prompt(demos, query)
        corrupted_demos = make_corrupted_demos(demos, rng, ensure_derangement=True)
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

        if clean_slot["target_id"] != corrupted_slot["target_id"]:
            log("target_id mismatch between clean and corrupted")
            log_file.close()
            return 1

        trials.append(
            {
                "clean_prefix_str": clean_prefix_str,
                "corrupted_prefix_str": corrupted_prefix_str,
                "target_id": clean_slot["target_id"],
                "target_token": clean_slot["target_token"],
            }
        )

    replace_vec = None
    if args.mode in ("clean_mean", "corrupted_mean"):
        prefixes = [
            item["clean_prefix_str"]
            if args.mode == "clean_mean"
            else item["corrupted_prefix_str"]
            for item in trials
        ]
        try:
            replace_vec = compute_mean_head_vec(
                prefixes,
                model,
                tokenizer,
                device,
                n_heads,
                head_dim,
                resid_dim,
                args.head,
                -1,
                target_module,
            )
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1
        if args.alpha is not None:
            replace_vec = replace_vec * args.alpha
        log(f"replace_vec prepared for mode={args.mode} (head_dim={head_dim})")

    results = []
    delta_logits = []
    max_abs_full_diff = 0.0

    for trial_idx, item in enumerate(trials):
        prefix_str = item["corrupted_prefix_str"]
        target_id = item["target_id"]
        target_token = item["target_token"]

        inputs = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]
        last_index = seq_len - 1

        with torch.inference_mode():
            outputs = model(**inputs)
        baseline_logits = outputs.logits[0, last_index]
        baseline_logit = baseline_logits[target_id].item()

        hook_fired = {"value": False}

        def hook_logger(message: str) -> None:
            hook_fired["value"] = True
            log(message)

        hook_mode = (
            "replace" if args.mode in ("clean_mean", "corrupted_mean") else args.mode
        )
        hook = make_cproj_head_replacer(
            layer_idx=args.layer,
            head_idx=args.head,
            token_idx=-1,
            mode=hook_mode,
            replace_vec=replace_vec,
            model_config=model_cfg,
            logger=hook_logger,
        )
        handle = target_module.register_forward_pre_hook(hook)
        with torch.inference_mode():
            outputs_patched = model(**inputs)
        handle.remove()

        patched_logits = outputs_patched.logits[0, last_index]
        patched_logit = patched_logits[target_id].item()
        delta_logit = patched_logit - baseline_logit
        delta_logits.append(delta_logit)

        full_diff = (patched_logits - baseline_logits).abs().max().item()
        if full_diff > max_abs_full_diff:
            max_abs_full_diff = full_diff

        results.append(
            {
                "trial_idx": trial_idx,
                "target_id": target_id,
                "target_token": target_token,
                "baseline_logit": baseline_logit,
                "patched_logit": patched_logit,
                "delta_logit": delta_logit,
                "hook_fired": bool(hook_fired["value"]),
                "layer": args.layer,
                "head": args.head,
                "mode": args.mode,
                "seq_len": seq_len,
            }
        )

    max_abs_delta_logit = max((abs(v) for v in delta_logits), default=0.0)
    mean_abs_delta_logit = mean_abs(delta_logits)

    self_replace_ok = False
    if args.mode == "self":
        self_replace_ok = max_abs_delta_logit < 1e-6 and max_abs_full_diff < 1e-6

    summary = {
        "n_trials": args.n_trials,
        "max_abs_delta_logit": max_abs_delta_logit,
        "mean_abs_delta_logit": mean_abs_delta_logit,
        "self_replace_ok": self_replace_ok,
        "max_abs_full_logit_diff": max_abs_full_diff,
    }

    out_path = os.path.join(artifacts_dir, "stepC_patch_sanity.json")
    save_json(out_path, {"summary": summary, "trials": results})

    log(
        "summary: "
        f"max_abs_delta_logit={max_abs_delta_logit:.6e} "
        f"mean_abs_delta_logit={mean_abs_delta_logit:.6e} "
        f"max_abs_full_logit_diff={max_abs_full_diff:.6e}"
    )
    log(f"self_replace_ok: {self_replace_ok}")
    log(f"saved results: {out_path}")
    log_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
