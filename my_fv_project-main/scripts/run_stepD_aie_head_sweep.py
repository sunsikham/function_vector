#!/usr/bin/env python3
"""STEP D: AIE head sweep using mean_activations replacement on corrupted prompts."""

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.corrupt import make_corrupted_demos
from fv.dataset_loader import load_pairs_antonym, sample_demos_and_query
from fv.hooks import get_out_proj_pre_hook_target
from fv.io import prepare_run_dirs, resolve_out_dir, save_csv, save_json
from fv.adapters import infer_head_dims, resolve_blocks
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.mean_activations import compute_mean_activations_ns
from fv.model_spec import get_model_spec
from fv.patch import make_out_proj_head_output_overrider
from fv.prompting import build_prompt_qa
from fv.slots import compute_query_predictive_slot


def make_logger(log_path: str):
    log_file = open(log_path, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    return log, log_file


def parse_layers(value: str):
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError("Invalid layer range")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def parse_heads(value: str, n_heads: int):
    if value == "all":
        return list(range(n_heads))
    heads = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        heads.append(int(part))
    return sorted(set(heads))


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def mean_abs(values):
    if not values:
        return 0.0
    return sum(abs(value) for value in values) / len(values)


def _make_demo_only_shuffle(demos, perm):
    outputs = [y for _x, y in demos]
    shuffled = [outputs[i] for i in perm]
    fixed_points = sum(1 for i, j in enumerate(perm) if i == j)
    shuffled_demos = [(demos[i][0], shuffled[i]) for i in range(len(demos))]
    return shuffled_demos, outputs, shuffled, fixed_points


def compute_trial_metrics(logits_base, logits_patch, target_id):
    import torch.nn.functional as F

    p_base = F.softmax(logits_base, dim=-1)[target_id].item()
    p_patch = F.softmax(logits_patch, dim=-1)[target_id].item()
    delta_p = p_patch - p_base

    logit_base = logits_base[target_id].item()
    logit_patch = logits_patch[target_id].item()
    delta_logit = logit_patch - logit_base

    logprob_base = F.log_softmax(logits_base, dim=-1)[target_id].item()
    logprob_patch = F.log_softmax(logits_patch, dim=-1)[target_id].item()
    delta_logprob = logprob_patch - logprob_base

    return {
        "p_base": p_base,
        "p_patch": p_patch,
        "delta_p": delta_p,
        "logit_base": logit_base,
        "logit_patch": logit_patch,
        "delta_logit": delta_logit,
        "logprob_base": logprob_base,
        "logprob_patch": logprob_patch,
        "delta_logprob": delta_logprob,
    }



def format_duration(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def compute_slot_with_fallback(
    prefix_str: str, full_str: str, tokenizer, log, add_special_tokens: bool = False
):
    try:
        return compute_query_predictive_slot(
            prefix_str,
            full_str,
            tokenizer,
            add_special_tokens=add_special_tokens,
        )
    except ValueError as exc:
        message = str(exc)
        if "Target id mismatch" not in message:
            raise
        if prefix_str.endswith(" "):
            raise
        trimmed_prefix = prefix_str.rstrip(" ")
        if trimmed_prefix == prefix_str:
            raise
        log("retrying slot computation with trimmed prefix space")
        return compute_query_predictive_slot(
            trimmed_prefix,
            full_str,
            tokenizer,
            add_special_tokens=add_special_tokens,
        )


def check_successful_icl(
    model, tokenizer, device, prefix_str: str, target_id: int, tok_add_special: bool
):
    import torch

    inputs = tokenizer(
        prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    last_index = inputs["input_ids"].shape[1] - 1
    with torch.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits[0, last_index]
    pred_id = int(torch.argmax(logits).item())
    p_target = torch.softmax(logits.float(), dim=-1)[target_id].item()
    success = pred_id == target_id
    return success, p_target


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP D AIE head sweep.")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument(
        "--dataset_path",
        default="datasets/processed/antonym.json",
        help="Path to antonym dataset JSON",
    )
    parser.add_argument(
        "--model_spec",
        default="gpt2",
        help="Model spec name for adapter resolution (default: gpt2)",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["fp16", "bf16", "fp32"],
        help="Model dtype (default: fp16 on cuda else fp32)",
    )
    parser.add_argument(
        "--quant",
        default="auto",
        choices=["auto", "none", "4bit", "8bit"],
        help="Quantization mode (default: auto)",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional HF device_map (default: None or 'auto')",
    )
    parser.add_argument(
        "--layers",
        default="0",
        help="Layer list (examples: --layers all, --layers 0,1,2,3)",
    )
    parser.add_argument("--heads", default="all", help="Head list or 'all' (default: all)")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials (default: 20)")
    parser.add_argument(
        "--n_icl_examples",
        type=int,
        default=3,
        help="Number of ICL demos per prompt (default: 3)",
    )
    parser.add_argument(
        "--n_mean_trials",
        type=int,
        default=None,
        help="Trials for mean_activations (default: n_trials)",
    )
    parser.add_argument(
        "--successful_icl_only",
        type=int,
        default=1,
        help="Keep only successful ICL trials (default: 1)",
    )
    parser.add_argument(
        "--max_trial_attempts",
        type=int,
        default=None,
        help="Max attempts when filtering successful ICL (default: n_trials*50)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--shuffle_labels",
        type=int,
        default=0,
        help="Shuffle demo labels only (default: 0)",
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
    parser.add_argument(
        "--save_trials",
        type=int,
        default=0,
        help="Save trial-level CSV (default: 0)",
    )
    parser.add_argument(
        "--score_key",
        default="mean_delta_p",
        help="Score key for ranking (default: mean_delta_p)",
    )
    args = parser.parse_args()

    if args.n_trials < 1:
        print("n_trials must be >= 1")
        return 1
    if args.n_icl_examples < 1:
        print("n_icl_examples must be >= 1")
        return 1
    if args.n_mean_trials is None:
        args.n_mean_trials = args.n_trials
    if args.n_mean_trials < 1:
        print("n_mean_trials must be >= 1")
        return 1
    if args.successful_icl_only not in (0, 1):
        print("successful_icl_only must be 0 or 1")
        return 1
    if args.max_trial_attempts is None:
        args.max_trial_attempts = args.n_trials * 50

    run_info = prepare_run_dirs(args.run_id)
    if args.out_dir:
        artifacts_dir = resolve_out_dir(args.out_dir)
    else:
        artifacts_dir = run_info["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(run_info["logs_dir"], "stepD_aie.log")
    log, log_file = make_logger(log_path)

    log("stepD AIE head sweep start")
    log(f"run_id: {run_info['run_id']}")
    log(f"artifacts_dir: {artifacts_dir}")
    log(f"log_path: {log_path}")
    log(f"model: {args.model}")
    log(f"dataset_path: {args.dataset_path}")
    log(f"model_spec: {args.model_spec}")
    log(f"layers: {args.layers}")
    log(f"heads: {args.heads}")
    log(f"n_trials: {args.n_trials}")
    log(f"n_icl_examples: {args.n_icl_examples}")
    log(f"n_mean_trials: {args.n_mean_trials}")
    log(f"successful_icl_only: {args.successful_icl_only}")
    log(f"max_trial_attempts: {args.max_trial_attempts}")
    log(f"seed: {args.seed}")
    log(f"score_key: {args.score_key}")
    log(f"shuffle_labels: {args.shuffle_labels}")

    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - runtime import check
        log(f"Failed to import required libraries: {exc}")
        log_file.close()
        return 1

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype is None:
        args.dtype = "fp16" if args.device == "cuda" else "fp32"

    try:
        spec = get_model_spec(args.model_spec)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1
    tok_add_special = bool(spec.prepend_bos)
    log(f"tok_add_special: {tok_add_special}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.shuffle_labels:
        if args.successful_icl_only:
            log(
                "shuffle_labels=True -> forcing successful_icl_only=0 "
                "(shuffled-label control)"
            )
        args.successful_icl_only = 0

    try:
        pairs = load_pairs_antonym(args.dataset_path, canonical_by_input=True)
    except Exception as exc:
        log(f"Failed to load dataset: {exc}")
        log_file.close()
        return 1

    try:
        loader_device = None if args.device_map else args.device
        model, tokenizer, diagnostics = load_hf_model_and_tokenizer(
            model_name=args.model,
            model_spec=args.model_spec,
            device=loader_device,
            dtype=args.dtype,
            quant=args.quant,
            device_map=args.device_map,
        )
        log(
            "hf_loader diagnostics: "
            + " ".join(
                f"{key}={value}" for key, value in diagnostics.items()
            )
        )
    except Exception as exc:  # pragma: no cover - runtime load check
        log(f"Failed to load model '{args.model}': {exc}")
        log_file.close()
        return 1

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    device = torch.device(args.device)
    resolved_quant = diagnostics.get("resolved_quant") if diagnostics else None
    if args.device_map:
        log("device_map provided; skipping model.to(device)")
        try:
            device = next(model.parameters()).device
        except StopIteration:
            pass
    elif resolved_quant in {"4bit", "8bit"}:
        log("quantized model; skipping model.to(device)")
    else:
        model.to(device)
    model.eval()
    log(
        "patched run uses manual_matmul_override "
        f"(resolved_quant={resolved_quant})"
    )

    try:
        dims = infer_head_dims(model, spec_name=args.model_spec)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    n_heads = int(dims["n_heads"])
    head_dim = int(dims["head_dim"])
    resid_dim = int(dims["hidden_size"])

    try:
        blocks = resolve_blocks(model, spec, logger=log)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    layer_count = len(blocks)
    if layer_count == 0:
        log("No layers available in resolved blocks")
        log_file.close()
        return 1
    model_cfg = {
        "n_heads": n_heads,
        "head_dim": head_dim,
        "resid_dim": resid_dim,
        "n_layers": layer_count,
    }
    if args.layers.strip().lower() == "all":
        layers = list(range(layer_count))
        log(f"[StepD] layers=all resolved to 0..{layer_count - 1} (n={layer_count})")
    else:
        try:
            layers = parse_layers(args.layers)
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1
        if not layers:
            log("No layers selected")
            log_file.close()
            return 1
        min_layer = min(layers)
        max_layer = max(layers)
        if min_layer < 0 or max_layer >= layer_count:
            log(
                "Layer index out of range: "
                f"allowed=[0, {layer_count - 1}] got={layers}"
            )
            log_file.close()
            return 1
        log(f"[StepD] sweeping layers: {min_layer}..{max_layer} (n={len(layers)})")

    try:
        heads = parse_heads(args.heads, n_heads)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1
    if not heads:
        log("No heads selected")
        log_file.close()
        return 1
    for head in heads:
        if head < 0 or head >= n_heads:
            log("Head index out of range")
            log_file.close()
            return 1

    if len(pairs) < args.n_icl_examples + 1:
        log(
            "Not enough dataset pairs for requested demos + query: "
            f"pairs={len(pairs)} n_icl_examples={args.n_icl_examples}"
        )
        log_file.close()
        return 1

    trials = []
    attempts = 0
    kept = 0
    p_targets = []

    if args.shuffle_labels not in (0, 1):
        log("shuffle_labels must be 0 or 1")
        log_file.close()
        return 1

    if args.successful_icl_only:
        while len(trials) < args.n_trials:
            if attempts >= args.max_trial_attempts:
                log(
                    "successful ICL이 너무 적으니 n_trials 줄이거나 "
                    "max_trial_attempts 늘리거나 successful_icl_only=0으로 끄라"
                )
                log_file.close()
                return 1
            attempt_idx = attempts
            attempts += 1
            demos_orig, query = sample_demos_and_query(
                pairs, args.n_icl_examples, seed=args.seed + attempt_idx
            )
            demo_perm = None
            demo_outputs_before = None
            demo_outputs_after = None
            demo_fixed_points = None
            if args.shuffle_labels:
                rng = random.Random(args.seed + attempt_idx)
                demo_perm = list(range(len(demos_orig)))
                rng.shuffle(demo_perm)
                demos, demo_outputs_before, demo_outputs_after, demo_fixed_points = (
                    _make_demo_only_shuffle(demos_orig, demo_perm)
                )
                if args.n_icl_examples == 1 and attempt_idx == 0:
                    log(f"demo_shuffle fixed_points={demo_fixed_points}")
            else:
                demos = demos_orig
            clean_prefix_str, clean_full_str = build_prompt_qa(demos, query)

            if attempts == 1:
                log(f"n_pairs_loaded: {len(pairs)}")
                log(f"n_icl_examples: {args.n_icl_examples}")
                log(f"example query: input='{query[0]}' output='{query[1]}'")
                log(f"prefix_endswith_A_space: {clean_prefix_str.endswith('A: ')}")

            try:
                clean_slot = compute_slot_with_fallback(
                    clean_prefix_str,
                    clean_full_str,
                    tokenizer,
                    log,
                    add_special_tokens=tok_add_special,
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1

            success, p_target = check_successful_icl(
                model,
                tokenizer,
                device,
                clean_prefix_str,
                clean_slot["target_id"],
                tok_add_special,
            )
            if not success:
                continue

            corrupted_demos = make_corrupted_demos(
                demos_orig, random.Random(args.seed + attempt_idx), ensure_derangement=True
            )
            if args.shuffle_labels:
                corrupted_demos, _, _, _ = _make_demo_only_shuffle(
                    corrupted_demos, demo_perm
                )
            corrupted_prefix_str, corrupted_full_str = build_prompt_qa(
                corrupted_demos, query
            )
            try:
                corrupted_slot = compute_slot_with_fallback(
                    corrupted_prefix_str,
                    corrupted_full_str,
                    tokenizer,
                    log,
                    add_special_tokens=tok_add_special,
                )
            except ValueError as exc:
                log(str(exc))
                log_file.close()
                return 1

            if clean_slot["target_id"] != corrupted_slot["target_id"]:
                log("target_id mismatch between clean and corrupted")
                log_file.close()
                return 1

            kept += 1
            p_targets.append(p_target)
            trials.append(
                {
                    "trial_idx": kept - 1,
                    "clean_prefix_str": clean_prefix_str,
                    "corrupted_prefix_str": corrupted_prefix_str,
                    "target_id": clean_slot["target_id"],
                    "target_token": clean_slot["target_token"],
                    "demo_perm": demo_perm,
                    "demo_fixed_points": demo_fixed_points,
                    "demo_outputs_before": demo_outputs_before,
                    "demo_outputs_after": demo_outputs_after,
                }
            )
    else:
        for trial_idx in range(args.n_trials):
            demos_orig, query = sample_demos_and_query(
                pairs, args.n_icl_examples, seed=args.seed + trial_idx
            )
            demo_perm = None
            demo_outputs_before = None
            demo_outputs_after = None
            demo_fixed_points = None
            if args.shuffle_labels:
                rng = random.Random(args.seed + trial_idx)
                demo_perm = list(range(len(demos_orig)))
                rng.shuffle(demo_perm)
                demos, demo_outputs_before, demo_outputs_after, demo_fixed_points = (
                    _make_demo_only_shuffle(demos_orig, demo_perm)
                )
                if args.n_icl_examples == 1 and trial_idx == 0:
                    log(f"demo_shuffle fixed_points={demo_fixed_points}")
            else:
                demos = demos_orig
            clean_prefix_str, clean_full_str = build_prompt_qa(demos, query)
            corrupted_demos = make_corrupted_demos(
                demos_orig, random.Random(args.seed + trial_idx), ensure_derangement=True
            )
            if args.shuffle_labels:
                corrupted_demos, _, _, _ = _make_demo_only_shuffle(
                    corrupted_demos, demo_perm
                )
            corrupted_prefix_str, corrupted_full_str = build_prompt_qa(
                corrupted_demos, query
            )

            if trial_idx == 0:
                log(f"n_pairs_loaded: {len(pairs)}")
                log(f"n_icl_examples: {args.n_icl_examples}")
                log(f"example query: input='{query[0]}' output='{query[1]}'")
                log(f"prefix_endswith_A_space: {clean_prefix_str.endswith('A: ')}")

            try:
                clean_slot = compute_slot_with_fallback(
                    clean_prefix_str,
                    clean_full_str,
                    tokenizer,
                    log,
                    add_special_tokens=tok_add_special,
                )
                corrupted_slot = compute_slot_with_fallback(
                    corrupted_prefix_str,
                    corrupted_full_str,
                    tokenizer,
                    log,
                    add_special_tokens=tok_add_special,
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
                    "trial_idx": trial_idx,
                    "clean_prefix_str": clean_prefix_str,
                    "corrupted_prefix_str": corrupted_prefix_str,
                    "target_id": clean_slot["target_id"],
                    "target_token": clean_slot["target_token"],
                    "demo_perm": demo_perm,
                    "demo_fixed_points": demo_fixed_points,
                    "demo_outputs_before": demo_outputs_before,
                    "demo_outputs_after": demo_outputs_after,
                }
            )

        attempts = args.n_trials
        kept = args.n_trials

    kept_ratio = kept / attempts if attempts else 0.0
    p_target_mean = mean(p_targets) if p_targets else 0.0
    log(
        f"trial_sampling_done attempts={attempts} kept={kept} kept_ratio={kept_ratio:.4f}"
    )
    if p_targets:
        log(f"p_target_mean={p_target_mean:.6f}")
    else:
        log("p_target_mean=n/a")

    layer_modules = {}
    for layer in layers:
        try:
            target_module, _target_name = get_out_proj_pre_hook_target(
                model, layer, spec_name=args.model_spec, logger=log
            )
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1
        layer_modules[layer] = target_module

    log("computing mean_activations")
    try:
        mean_acts, dummy_labels, slot_index_map = compute_mean_activations_ns(
            model=model,
            tokenizer=tokenizer,
            layer_modules=layer_modules,
            pairs=pairs,
            n_icl_examples=args.n_icl_examples,
            n_mean_trials=args.n_mean_trials,
            model_cfg=model_cfg,
            seed=args.seed,
            tok_add_special=tok_add_special,
            device=device,
            shuffle_labels=bool(args.shuffle_labels),
            logger=log,
        )
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    slot_q = slot_index_map.get("QUERY_PRED")
    if slot_q is None:
        log("QUERY_PRED missing from slot_index_map")
        log_file.close()
        return 1
    log(f"patch config: token_idx=-1 (last token), slot_q={slot_q} (QUERY_PRED)")
    log(
        "mean_activations: "
        f"shape={tuple(mean_acts.shape)} n_slots={len(dummy_labels)} "
        f"slot_q={slot_q} label={dummy_labels[slot_q]}"
    )

    mean_acts_path = os.path.join(artifacts_dir, "mean_activations.pt")
    torch.save(mean_acts.cpu(), mean_acts_path)
    log(f"saved mean_activations: {mean_acts_path}")

    mean_meta_path = os.path.join(artifacts_dir, "mean_activations_meta.json")
    save_json(
        mean_meta_path,
        {
            "n_mean_trials": args.n_mean_trials,
            "n_icl_examples": args.n_icl_examples,
            "seed": args.seed,
            "layers": layers,
            "n_layers": layer_count,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "dummy_labels": dummy_labels,
            "slot_index_map": slot_index_map,
            "slot_query_pred": slot_q,
        },
    )
    log(f"saved mean_activations meta: {mean_meta_path}")

    stepd_filter_path = os.path.join(artifacts_dir, "stepD_success_filter.json")
    save_json(
        stepd_filter_path,
        {
            "successful_icl_only": args.successful_icl_only,
            "max_trial_attempts": args.max_trial_attempts,
            "attempts": attempts,
            "kept": kept,
            "kept_ratio": kept_ratio,
            "seed": args.seed,
            "n_icl_examples": args.n_icl_examples,
        },
    )
    log(f"saved success filter: {stepd_filter_path}")

    run_meta_path = os.path.join(artifacts_dir, "run_meta.json")
    save_json(
        run_meta_path,
        {
            "shuffle_labels": bool(args.shuffle_labels),
            "shuffle_derangement": False,
            "successful_icl_only_effective": args.successful_icl_only,
            "seed_global": args.seed,
        },
    )
    log(f"saved run meta: {run_meta_path}")

    scores = []
    trials_rows = []
    trial_metrics_path = os.path.join(artifacts_dir, "trial_metrics.jsonl")
    trial_metrics_file = open(trial_metrics_path, "w", encoding="utf-8")

    log("starting AIE sweep")
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    pairs = [(layer, head) for layer in layers for head in heads]
    total = len(pairs)
    log_every = max(1, total // 100)
    start_time = time.time()
    pbar = (
        tqdm(pairs, total=total, desc="StepD layer×head", unit="head")
        if tqdm is not None
        else None
    )

    for idx_pair, (layer, head) in enumerate(pairs, start=1):
        if pbar is not None:
            pbar.set_postfix({"layer": layer, "head": head, "trials": args.n_trials})
        if idx_pair == 1 or idx_pair % log_every == 0 or idx_pair == total:
            elapsed = time.time() - start_time
            rate = idx_pair / elapsed if elapsed > 0 else 0.0
            remaining = total - idx_pair
            eta = remaining / rate if rate > 0 else 0.0
            percent = (idx_pair / total) * 100 if total else 100.0
            log(
                "[StepD] "
                f"{percent:.1f}% ({idx_pair}/{total}) "
                f"layer={layer} head={head} "
                f"elapsed={format_duration(elapsed)} "
                f"ETA={format_duration(eta)}"
            )

        metric_lists = {
            "p_base": [],
            "p_patch": [],
            "delta_p": [],
            "logit_base": [],
            "logit_patch": [],
            "delta_logit": [],
            "logprob_base": [],
            "logprob_patch": [],
            "delta_logprob": [],
        }
        nonzero = False
        replace_vec = mean_acts[layer, head, slot_q]
        hook = make_out_proj_head_output_overrider(
            layer_idx=layer,
            head_idx=head,
            token_idx=-1,
            mode="replace",
            replace_vec=replace_vec,
            model_config=model_cfg,
            resolved_quant=resolved_quant,
            logger=log,
        )
        for trial in trials:
            prefix_str = trial["corrupted_prefix_str"]
            target_id = trial["target_id"]

            inputs = tokenizer(
                prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            baseline_logits = outputs.logits[0, -1]
            if idx_pair == 1 and trial["trial_idx"] == 0:
                target_token = tokenizer.convert_ids_to_tokens(target_id)
                prompt_tail = repr(prefix_str[-20:])
                log(
                    "target_debug: "
                    f"prompt_tail={prompt_tail} "
                    f"target_token={target_token} target_id={target_id}"
                )

            handle = layer_modules[layer].register_forward_hook(hook)
            with torch.inference_mode():
                outputs_patched = model(**inputs)
            handle.remove()

            patched_logits = outputs_patched.logits[0, -1]
            trial_metrics = compute_trial_metrics(
                baseline_logits, patched_logits, target_id
            )
            for key, value in trial_metrics.items():
                metric_lists[key].append(value)
            if abs(trial_metrics["delta_p"]) > 1e-12:
                nonzero = True

            trial_row = {
                "trial_idx": trial["trial_idx"],
                "layer": layer,
                "head": head,
                "target_id": target_id,
                "target_token": tokenizer.convert_ids_to_tokens(target_id),
                "seed_global": args.seed,
                "shuffle_labels": bool(args.shuffle_labels),
                "shuffle_derangement": False,
                "n_icl_examples": args.n_icl_examples,
                "demo_perm": trial.get("demo_perm"),
                "demo_fixed_points": trial.get("demo_fixed_points"),
                "demo_outputs_before": trial.get("demo_outputs_before"),
                "demo_outputs_after": trial.get("demo_outputs_after"),
                "p_base": trial_metrics["p_base"],
                "p_patch": trial_metrics["p_patch"],
                "delta_p": trial_metrics["delta_p"],
                "logit_base": trial_metrics["logit_base"],
                "logit_patch": trial_metrics["logit_patch"],
                "delta_logit": trial_metrics["delta_logit"],
                "logprob_base": trial_metrics["logprob_base"],
                "logprob_patch": trial_metrics["logprob_patch"],
                "delta_logprob": trial_metrics["delta_logprob"],
                "prompt_tail_repr": repr(prefix_str[-30:]),
                "prompt_ends_with_space": prefix_str.endswith(" "),
                "token_idx": -1,
                "slot_q": slot_q,
            }
            trial_metrics_file.write(json.dumps(trial_row, ensure_ascii=True) + "\n")

            if args.save_trials:
                trials_rows.append(
                    {
                        "trial_idx": trial["trial_idx"],
                        "layer": layer,
                        "head": head,
                        "target_id": target_id,
                        "p_base": trial_metrics["p_base"],
                        "p_patch": trial_metrics["p_patch"],
                        "delta_p": trial_metrics["delta_p"],
                        "logit_base": trial_metrics["logit_base"],
                        "logit_patch": trial_metrics["logit_patch"],
                        "delta_logit": trial_metrics["delta_logit"],
                        "logprob_base": trial_metrics["logprob_base"],
                        "logprob_patch": trial_metrics["logprob_patch"],
                        "delta_logprob": trial_metrics["delta_logprob"],
                    }
                )

        mean_act_norm = mean_acts[layer, head, slot_q].norm().item()
        mean_delta_p = mean(metric_lists["delta_p"])
        std_delta_p = std(metric_lists["delta_p"])
        mean_abs_delta_p = mean_abs(metric_lists["delta_p"])
        mean_p_base = mean(metric_lists["p_base"])
        std_p_base = std(metric_lists["p_base"])
        mean_p_patch = mean(metric_lists["p_patch"])
        std_p_patch = std(metric_lists["p_patch"])

        mean_delta_logit = mean(metric_lists["delta_logit"])
        std_delta_logit = std(metric_lists["delta_logit"])
        mean_abs_delta_logit = mean_abs(metric_lists["delta_logit"])
        mean_logit_base = mean(metric_lists["logit_base"])
        mean_logit_patch = mean(metric_lists["logit_patch"])

        mean_delta_logprob = mean(metric_lists["delta_logprob"])
        std_delta_logprob = std(metric_lists["delta_logprob"])
        mean_abs_delta_logprob = mean_abs(metric_lists["delta_logprob"])
        mean_logprob_base = mean(metric_lists["logprob_base"])
        mean_logprob_patch = mean(metric_lists["logprob_patch"])
        scores.append(
            {
                "layer": layer,
                "head": head,
                "n_trials": args.n_trials,
                "mean_delta_p": mean_delta_p,
                "std_delta_p": std_delta_p,
                "mean_abs_delta_p": mean_abs_delta_p,
                "mean_p_base": mean_p_base,
                "std_p_base": std_p_base,
                "mean_p_patch": mean_p_patch,
                "std_p_patch": std_p_patch,
                "mean_delta_logit": mean_delta_logit,
                "std_delta_logit": std_delta_logit,
                "mean_abs_delta_logit": mean_abs_delta_logit,
                "mean_logit_base": mean_logit_base,
                "mean_logit_patch": mean_logit_patch,
                "mean_delta_logprob": mean_delta_logprob,
                "std_delta_logprob": std_delta_logprob,
                "mean_abs_delta_logprob": mean_abs_delta_logprob,
                "mean_logprob_base": mean_logprob_base,
                "mean_logprob_patch": mean_logprob_patch,
                "any_nonzero": nonzero,
                "mean_act_norm": mean_act_norm,
                "clean_mean_norm": mean_act_norm,
            }
        )
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    if scores and args.score_key not in scores[0]:
        log(f"score_key not found: {args.score_key}")
        log_file.close()
        trial_metrics_file.close()
        return 1
    scores_sorted = sorted(scores, key=lambda row: row[args.score_key], reverse=True)
    log(f"top-10 by {args.score_key}:")
    for row in scores_sorted[:10]:
        log(
            "top: "
            f"layer={row['layer']} "
            f"head={row['head']} "
            f"{args.score_key}={row[args.score_key]:.6f}"
        )

    scores_path = os.path.join(artifacts_dir, "aie_scores.csv")
    save_csv(scores_path, scores_sorted)

    scores_json_path = os.path.join(artifacts_dir, "aie_scores.json")
    save_json(
        scores_json_path,
        {
            "meta": {
                "model": args.model,
                "model_spec": args.model_spec,
                "layers": layers,
                "heads": heads,
                "n_trials": args.n_trials,
                "n_icl_examples": args.n_icl_examples,
                "n_mean_trials": args.n_mean_trials,
                "seed": args.seed,
                "successful_icl_only": args.successful_icl_only,
                "attempts": attempts,
                "kept": kept,
                "kept_ratio": kept_ratio,
                "token_idx": -1,
                "slot_query_pred": slot_q,
                "slot_label": dummy_labels[slot_q],
                "score_key": args.score_key,
            },
            "scores": scores_sorted,
        },
    )

    log(f"saved scores: {scores_path}")
    log(f"saved scores json: {scores_json_path}")

    if args.save_trials:
        trials_path = os.path.join(artifacts_dir, "aie_trials.csv")
        save_csv(trials_path, trials_rows)
        log(f"saved trials: {trials_path}")

    trial_metrics_file.close()
    log(f"saved trial metrics: {trial_metrics_path}")

    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
