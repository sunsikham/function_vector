#!/usr/bin/env python3
"""STEP D: AIE head sweep using clean-mean replacement on corrupted prompts."""

import argparse
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
from fv.hooks import get_out_proj_pre_hook_target, reshape_resid_to_heads
from fv.io import prepare_run_dirs, resolve_out_dir, save_csv, save_json
from fv.adapters import infer_head_dims, resolve_blocks
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.model_spec import get_model_spec
from fv.patch import make_cproj_head_replacer
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


def compute_token_scores(logits, target_id):
    import torch

    logits32 = logits.float()
    logp_all = logits32 - torch.logsumexp(logits32, dim=-1, keepdim=True)
    logprob = logp_all[..., target_id]
    p = torch.exp(logprob)
    logit = logits32[..., target_id]
    return {"logit": logit, "p": p, "logprob": logprob}


def compute_clean_mean(
    prefixes,
    model,
    tokenizer,
    device,
    layers,
    n_heads,
    head_dim,
    resid_dim,
    layer_modules,
    log,
    add_special_tokens: bool = False,
):
    import torch

    means = {layer: torch.zeros((n_heads, head_dim), device=device) for layer in layers}
    counts = {layer: 0 for layer in layers}

    for layer in layers:
        state = {"current": None, "errors": []}

        def pre_hook(_module, inputs):
            tensor = inputs[0] if inputs else None
            try:
                reshaped = reshape_resid_to_heads(tensor, n_heads, head_dim, resid_dim)
            except ValueError as exc:
                state["errors"].append(str(exc))
                return
            seq_len = reshaped.shape[1]
            t_idx = seq_len - 1
            state["current"] = reshaped[:, t_idx, :, :].detach().mean(dim=0)

        handle = layer_modules[layer].register_forward_pre_hook(pre_hook)
        for prefix in prefixes:
            state["current"] = None
            inputs = tokenizer(
                prefix, return_tensors="pt", add_special_tokens=add_special_tokens
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                _ = model(**inputs)
            if state["errors"]:
                handle.remove()
                raise ValueError("; ".join(state["errors"]))
            if state["current"] is None:
                handle.remove()
                raise ValueError("Failed to capture head activations")
            means[layer] += state["current"]
            counts[layer] += 1
        handle.remove()
        log(
            f"clean_mean layer={layer} shape=({n_heads},{head_dim}) count={counts[layer]}"
        )

    for layer in layers:
        if counts[layer] == 0:
            raise ValueError("No clean_mean samples captured")
        means[layer] = means[layer] / counts[layer]

    return means


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
        "--cie_metric",
        default="logprob",
        choices=["logit", "p", "logprob"],
        help="Metric for mean_cie/std_cie (default: logprob)",
    )
    args = parser.parse_args()

    if args.n_trials < 1:
        print("n_trials must be >= 1")
        return 1
    if args.n_icl_examples < 1:
        print("n_icl_examples must be >= 1")
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
    log(f"successful_icl_only: {args.successful_icl_only}")
    log(f"max_trial_attempts: {args.max_trial_attempts}")
    log(f"seed: {args.seed}")
    log(f"cie_metric: {args.cie_metric}")

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

    try:
        dims = infer_head_dims(model, spec_name=args.model_spec)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    n_heads = int(dims["n_heads"])
    head_dim = int(dims["head_dim"])
    resid_dim = int(dims["hidden_size"])
    model_cfg = {"n_heads": n_heads, "head_dim": head_dim, "resid_dim": resid_dim}

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
            demos, query = sample_demos_and_query(
                pairs, args.n_icl_examples, seed=args.seed + attempt_idx
            )
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
                demos, random.Random(args.seed + attempt_idx), ensure_derangement=True
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
                }
            )
    else:
        for trial_idx in range(args.n_trials):
            demos, query = sample_demos_and_query(
                pairs, args.n_icl_examples, seed=args.seed + trial_idx
            )
            clean_prefix_str, clean_full_str = build_prompt_qa(demos, query)
            corrupted_demos = make_corrupted_demos(
                demos, random.Random(args.seed + trial_idx), ensure_derangement=True
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

    log("computing clean_mean")
    try:
        clean_mean = compute_clean_mean(
            [t["clean_prefix_str"] for t in trials],
            model,
            tokenizer,
            device,
            layers,
            n_heads,
            head_dim,
            resid_dim,
            layer_modules,
            log,
            add_special_tokens=tok_add_special,
        )
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    clean_mean_path = os.path.join(artifacts_dir, "clean_mean.pt")
    torch.save(
        {
            "layers": layers,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "model_spec": args.model_spec,
            "clean_mean": {layer: clean_mean[layer].cpu() for layer in layers},
        },
        clean_mean_path,
    )
    log(f"saved clean_mean: {clean_mean_path}")

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

    scores = []
    trials_rows = []

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

        cie_vals = {"logit": [], "p": [], "logprob": []}
        nonzero = False
        for trial in trials:
            prefix_str = trial["corrupted_prefix_str"]
            target_id = trial["target_id"]

            inputs = tokenizer(
                prefix_str, return_tensors="pt", add_special_tokens=tok_add_special
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            last_index = inputs["input_ids"].shape[1] - 1

            with torch.inference_mode():
                outputs = model(**inputs)
            baseline_logits = outputs.logits[0, last_index]
            baseline_logit = baseline_logits[target_id].item()
            baseline_scores = compute_token_scores(baseline_logits, target_id)

            replace_vec = clean_mean[layer][head]
            hook = make_cproj_head_replacer(
                layer_idx=layer,
                head_idx=head,
                token_idx=-1,
                mode="replace",
                replace_vec=replace_vec,
                model_config=model_cfg,
                logger=None,
            )
            handle = layer_modules[layer].register_forward_pre_hook(hook)
            with torch.inference_mode():
                outputs_patched = model(**inputs)
            handle.remove()

            patched_logits = outputs_patched.logits[0, last_index]
            patched_logit = patched_logits[target_id].item()
            patched_scores = compute_token_scores(patched_logits, target_id)
            delta_scores = {
                key: (patched_scores[key] - baseline_scores[key]).item()
                for key in baseline_scores
            }
            for key, value in delta_scores.items():
                cie_vals[key].append(value)
            if abs(delta_scores[args.cie_metric]) > 1e-12:
                nonzero = True

            if args.save_trials:
                trials_rows.append(
                    {
                        "trial_idx": trial["trial_idx"],
                        "layer": layer,
                        "head": head,
                        "baseline_logit": baseline_logit,
                        "patched_logit": patched_logit,
                        "cie": delta_scores[args.cie_metric],
                        "target_id": target_id,
                    }
                )

        clean_norm = clean_mean[layer][head].norm().item()
        mean_cie_logit = mean(cie_vals["logit"])
        std_cie_logit = std(cie_vals["logit"])
        mean_cie_p = mean(cie_vals["p"])
        std_cie_p = std(cie_vals["p"])
        mean_cie_logprob = mean(cie_vals["logprob"])
        std_cie_logprob = std(cie_vals["logprob"])
        if args.cie_metric == "logit":
            mean_cie = mean_cie_logit
            std_cie = std_cie_logit
            mean_abs_cie = mean_abs(cie_vals["logit"])
        elif args.cie_metric == "p":
            mean_cie = mean_cie_p
            std_cie = std_cie_p
            mean_abs_cie = mean_abs(cie_vals["p"])
        else:
            mean_cie = mean_cie_logprob
            std_cie = std_cie_logprob
            mean_abs_cie = mean_abs(cie_vals["logprob"])
        scores.append(
            {
                "layer": layer,
                "head": head,
                "n_trials": args.n_trials,
                "mean_cie": mean_cie,
                "std_cie": std_cie,
                "mean_abs_cie": mean_abs_cie,
                "mean_cie_logit": mean_cie_logit,
                "std_cie_logit": std_cie_logit,
                "mean_cie_p": mean_cie_p,
                "std_cie_p": std_cie_p,
                "mean_cie_logprob": mean_cie_logprob,
                "std_cie_logprob": std_cie_logprob,
                "any_nonzero": nonzero,
                "clean_mean_norm": clean_norm,
            }
        )
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()

    scores_sorted = sorted(scores, key=lambda row: row["mean_cie"], reverse=True)
    log("top-10 by mean_cie:")
    for row in scores_sorted[:10]:
        log(
            "top: "
            f"layer={row['layer']} "
            f"head={row['head']} "
            f"mean_cie={row['mean_cie']:.6f} "
            f"std_cie={row['std_cie']:.6f}"
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
                "seed": args.seed,
                "successful_icl_only": args.successful_icl_only,
                "attempts": attempts,
                "kept": kept,
                "kept_ratio": kept_ratio,
                "token_idx": -1,
                "measure": f"{args.cie_metric}[target_id]",
                "cie_metric": args.cie_metric,
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

    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
