#!/usr/bin/env python3
"""STEP E: Build top-k FV from StepD and run injection eval."""

import argparse
import csv
import os
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.adapters import infer_head_dims, resolve_attn, resolve_blocks, resolve_out_proj
from fv.corrupt import make_corrupted_demos
from fv.dataset_loader import load_pairs_antonym, sample_demos_and_query
from fv.intervene import make_residual_injection_hook
from fv.io import prepare_run_dirs, resolve_out_dir, save_json
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.model_spec import get_model_spec
from fv.prompting import build_prompt_qa
from fv.slots import compute_query_predictive_slot


def make_logger(log_path: str):
    log_file = open(log_path, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    return log, log_file


def parse_layers(value: str, available_layers):
    if value == "all":
        return sorted(available_layers)
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


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std(values):
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def build_fv_global_resid(
    model,
    spec,
    top_heads,
    clean_mean,
    head_dim: int,
    hidden_size: int,
    device,
    log,
):
    import torch

    fv_global = torch.zeros((hidden_size,), dtype=torch.float32, device=device)
    blocks = resolve_blocks(model, spec, logger=log)
    out_proj_by_layer = {}

    for entry in top_heads:
        layer = entry["layer"]
        head = entry["head"]
        if layer not in out_proj_by_layer:
            attn = resolve_attn(blocks[layer], spec, logger=log)
            out_proj_by_layer[layer] = resolve_out_proj(attn, spec, logger=log)
        out_proj = out_proj_by_layer[layer]

        try:
            param = next(out_proj.parameters())
            out_dtype = param.dtype
            out_device = param.device
        except StopIteration:
            out_dtype = torch.float32
            out_device = device

        x_pre = torch.zeros((1, 1, hidden_size), dtype=out_dtype, device=out_device)
        head_vec = clean_mean[layer][head].to(device=out_device, dtype=out_dtype)
        start = head * head_dim
        end = start + head_dim
        x_pre[0, 0, start:end] = head_vec

        with torch.inference_mode():
            out = out_proj(x_pre)
        if isinstance(out, tuple):
            out = out[0]
        fv_global += out.reshape(-1).to(dtype=torch.float32, device=device)

    return fv_global


def compute_slot_with_fallback(prefix_str: str, full_str: str, tokenizer, log):
    try:
        return compute_query_predictive_slot(prefix_str, full_str, tokenizer)
    except ValueError as exc:
        message = str(exc)
        if "Target id mismatch" not in message:
            raise
        trimmed_prefix = prefix_str.rstrip(" ")
        if trimmed_prefix == prefix_str:
            raise
        log("retrying slot computation with trimmed prefix space")
        return compute_query_predictive_slot(trimmed_prefix, full_str, tokenizer)


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP E top-k FV + eval.")
    parser.add_argument("--run_id_stepD", required=True, help="StepD run_id")
    parser.add_argument("--k", type=int, default=20, help="Top-k heads (default: 20)")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument(
        "--metric",
        default="mean_cie",
        choices=["mean_cie", "mean_abs_cie"],
        help="AIE ranking score column (default: mean_cie)",
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
        "--dataset_path",
        default="datasets/processed/antonym.json",
        help="Path to antonym dataset JSON",
    )
    parser.add_argument(
        "--model_spec",
        default="gpt2",
        help="Model spec name for adapter resolution (default: gpt2)",
    )
    parser.add_argument("--layers", default="all", help="Layer filter (default: all)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Injection scale (default: 1.0)")
    parser.add_argument(
        "--n_eval_trials",
        type=int,
        default=20,
        help="Evaluation trials (default: 20)",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip StepE eval (no stepE_eval.json)",
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
    args = parser.parse_args()

    if args.k < 1:
        print("k must be >= 1")
        return 1
    if args.n_eval_trials < 1:
        print("n_eval_trials must be >= 1")
        return 1

    run_info = prepare_run_dirs(args.run_id)
    if args.out_dir:
        artifacts_dir = resolve_out_dir(args.out_dir)
    else:
        artifacts_dir = run_info["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(run_info["logs_dir"], "stepE.log")
    log, log_file = make_logger(log_path)

    log("stepE start")
    log(f"run_id: {run_info['run_id']}")
    log(f"artifacts_dir: {artifacts_dir}")
    log(f"log_path: {log_path}")
    log(f"run_id_stepD: {args.run_id_stepD}")
    log(f"model: {args.model}")
    log(f"metric: {args.metric}")
    log(f"dataset_path: {args.dataset_path}")
    log(f"model_spec: {args.model_spec}")
    log(f"k: {args.k}")
    log(f"layers: {args.layers}")
    log(f"alpha: {args.alpha}")
    log(f"n_eval_trials: {args.n_eval_trials}")
    log(f"skip_eval: {args.skip_eval}")
    log(f"seed: {args.seed}")

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
        pairs = load_pairs_antonym(args.dataset_path, canonical_by_input=True)
    except Exception as exc:
        log(f"Failed to load dataset: {exc}")
        log_file.close()
        return 1

    try:
        spec = get_model_spec(args.model_spec)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    stepd_dir = resolve_out_dir(os.path.join("runs", args.run_id_stepD, "artifacts"))
    scores_path = os.path.join(stepd_dir, "aie_scores.csv")
    clean_mean_path = os.path.join(stepd_dir, "clean_mean.pt")
    if not os.path.exists(scores_path):
        log(f"Missing aie_scores.csv: {scores_path}")
        log_file.close()
        return 1
    if not os.path.exists(clean_mean_path):
        log(f"Missing clean_mean.pt: {clean_mean_path}")
        log_file.close()
        return 1

    scores = []
    with open(scores_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            log("aie_scores.csv missing header row")
            log_file.close()
            return 1
        if args.metric not in reader.fieldnames:
            log(
                "Requested metric not found in aie_scores.csv: "
                f"metric='{args.metric}' available={reader.fieldnames}"
            )
            log_file.close()
            return 1
        for row in reader:
            scores.append(row)

    available_layers = sorted({int(row["layer"]) for row in scores})
    try:
        layer_filter = set(parse_layers(args.layers, available_layers))
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1
    layer_filter &= set(available_layers)
    if not layer_filter:
        log("No layers left after filtering")
        log_file.close()
        return 1

    filtered = [
        row
        for row in scores
        if int(row["layer"]) in layer_filter
    ]
    filtered.sort(key=lambda r: float(r[args.metric]), reverse=True)
    top_rows = filtered[: args.k]
    if not top_rows:
        log("No heads found for top-k selection")
        log_file.close()
        return 1

    top_heads = []
    for row in top_rows:
        top_heads.append(
            {
                "layer": int(row["layer"]),
                "head": int(row["head"]),
                "mean_cie": float(row["mean_cie"]),
                "mean_abs_cie": float(row["mean_abs_cie"]),
                "std_cie": float(row["std_cie"]),
                "n_trials": int(row["n_trials"]),
                "rank_score": float(row[args.metric]),
            }
        )

    top_heads_path = os.path.join(artifacts_dir, "top_heads.json")
    save_json(
        top_heads_path,
        {
            "meta": {
                "source_run_id": args.run_id_stepD,
                "k": args.k,
                "model": args.model,
                "model_spec": args.model_spec,
                "rank_metric": args.metric,
            },
            "heads": top_heads,
        },
    )
    log(f"saved top heads: {top_heads_path}")

    clean_mean_blob = torch.load(clean_mean_path, map_location="cpu")
    clean_mean = clean_mean_blob["clean_mean"]
    n_heads_blob = int(clean_mean_blob["n_heads"])
    head_dim_blob = int(clean_mean_blob["head_dim"])
    resid_dim_blob = int(clean_mean_blob["resid_dim"])

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

    if (n_heads, head_dim, resid_dim) != (n_heads_blob, head_dim_blob, resid_dim_blob):
        log(
            "clean_mean dims do not match model config: "
            f"clean_mean=({n_heads_blob},{head_dim_blob},{resid_dim_blob}) "
            f"model=({n_heads},{head_dim},{resid_dim})"
        )
        log_file.close()
        return 1

    fv_by_layer = {layer: torch.zeros((resid_dim,), dtype=torch.float32) for layer in layer_filter}
    for entry in top_heads:
        layer = entry["layer"]
        head = entry["head"]
        if layer not in fv_by_layer:
            continue
        head_vec = clean_mean[layer][head].to(dtype=torch.float32)
        resid_vec = torch.zeros((resid_dim,), dtype=torch.float32)
        start = head * head_dim
        end = start + head_dim
        resid_vec[start:end] = head_vec
        fv_by_layer[layer] += resid_vec

    fv_by_layer_path = os.path.join(artifacts_dir, "fv_by_layer.pt")
    torch.save(
        {
            "n_heads": n_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "selected_k": args.k,
            "alpha_default": args.alpha,
            "model_spec": args.model_spec,
            "fv_by_layer": {layer: fv_by_layer[layer] for layer in fv_by_layer},
        },
        fv_by_layer_path,
    )
    log(f"saved fv_by_layer: {fv_by_layer_path}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    try:
        blocks = resolve_blocks(model, spec, logger=log)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    for layer in fv_by_layer:
        if layer < 0 or layer >= len(blocks):
            log("Layer index out of range")
            log_file.close()
            return 1

    fv_global_resid = build_fv_global_resid(
        model=model,
        spec=spec,
        top_heads=top_heads,
        clean_mean=clean_mean,
        head_dim=head_dim,
        hidden_size=resid_dim,
        device=device,
        log=log,
    )
    if fv_global_resid.shape[-1] != resid_dim:
        log(
            "fv_global_resid size mismatch: "
            f"got={fv_global_resid.shape[-1]} expected={resid_dim}"
        )
        log_file.close()
        return 1
    log("built fv_global_resid via out_proj forward")

    fv_global_path = os.path.join(artifacts_dir, "fv_global_resid.pt")
    torch.save(
        {
            "fv_global_resid": fv_global_resid.detach().cpu(),
            "resid_dim": resid_dim,
            "head_dim": head_dim,
            "n_heads": n_heads,
        },
        fv_global_path,
    )
    log(f"saved fv_global_resid: {fv_global_path}")

    fv_global_meta_path = os.path.join(artifacts_dir, "fv_global_resid_meta.json")
    save_json(
        fv_global_meta_path,
        {
            "spec_name": spec.name,
            "model": args.model,
            "run_id_stepD": args.run_id_stepD,
            "heads": top_heads,
            "token_position_rule": "t_idx = seq_len - 1 (last token of prefix)",
        },
    )
    log(f"saved fv_global_resid metadata: {fv_global_meta_path}")

    if args.skip_eval:
        log("skip_eval enabled; skipping stepE_eval.json")
        log_file.close()
        return 0

    if len(pairs) < 4:
        log("Not enough dataset pairs for requested demos + query.")
        log_file.close()
        return 1

    trials = []
    for trial_idx in range(args.n_eval_trials):
        demos, query = sample_demos_and_query(pairs, 3, seed=args.seed + trial_idx)
        clean_prefix_str, clean_full_str = build_prompt_qa(demos, query)
        corrupted_demos = make_corrupted_demos(
            demos, random.Random(args.seed + trial_idx), ensure_derangement=True
        )
        corrupted_prefix_str, corrupted_full_str = build_prompt_qa(
            corrupted_demos, query
        )

        if trial_idx == 0:
            log(f"n_pairs_loaded: {len(pairs)}")
            log("n_icl_examples: 3")
            log(f"example query: input='{query[0]}' output='{query[1]}'")
            log(f"prefix_endswith_A_space: {clean_prefix_str.endswith('A: ')}")

        try:
            clean_slot = compute_slot_with_fallback(
                clean_prefix_str, clean_full_str, tokenizer, log
            )
            corrupted_slot = compute_slot_with_fallback(
                corrupted_prefix_str, corrupted_full_str, tokenizer, log
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
                "corrupted_prefix_str": corrupted_prefix_str,
                "target_id": clean_slot["target_id"],
            }
        )

    per_layer = {layer: [] for layer in fv_by_layer}
    all_deltas = []

    log("running eval")
    for trial in trials:
        prefix_str = trial["corrupted_prefix_str"]
        target_id = trial["target_id"]

        inputs = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        last_index = inputs["input_ids"].shape[1] - 1

        with torch.inference_mode():
            outputs = model(**inputs)
        baseline_logits = outputs.logits[0, last_index]
        baseline_logit = baseline_logits[target_id].item()

        for layer, fv_vec in fv_by_layer.items():
            injection_state = {
                "fv": fv_vec.to(device=device, dtype=baseline_logits.dtype),
                "alpha": args.alpha,
                "batch_indices": None,
                "last_indices": None,
            }
            batch_indices = torch.tensor([0], device=device)
            last_indices = torch.tensor([last_index], device=device)
            injection_state["batch_indices"] = batch_indices
            injection_state["last_indices"] = last_indices

            inject_hook = make_residual_injection_hook(injection_state)
            block = blocks[layer]
            handle = block.register_forward_hook(inject_hook)
            with torch.inference_mode():
                outputs_fv = model(**inputs)
            handle.remove()

            patched_logits = outputs_fv.logits[0, last_index]
            patched_logit = patched_logits[target_id].item()
            delta_logit = patched_logit - baseline_logit
            per_layer[layer].append(delta_logit)
            all_deltas.append(delta_logit)

    per_layer_summary = {}
    for layer, deltas in per_layer.items():
        per_layer_summary[str(layer)] = {
            "mean_delta_logit": mean(deltas),
            "std_delta_logit": std(deltas),
            "any_nonzero": any(abs(v) > 1e-12 for v in deltas),
            "n": len(deltas),
        }

    overall = {
        "mean_delta_logit": mean(all_deltas),
        "std_delta_logit": std(all_deltas),
        "any_nonzero": any(abs(v) > 1e-12 for v in all_deltas),
    }

    eval_path = os.path.join(artifacts_dir, "stepE_eval.json")
    save_json(
        eval_path,
        {
            "meta": {
                "model": args.model,
                "model_spec": args.model_spec,
                "run_id_stepD": args.run_id_stepD,
                "k": args.k,
                "layers": sorted(layer_filter),
                "alpha": args.alpha,
                "n_eval_trials": args.n_eval_trials,
                "seed": args.seed,
                "token_idx": -1,
                "measure": "logit[target_id]",
            },
            "per_layer": per_layer_summary,
            "overall": overall,
        },
    )

    log(f"saved eval: {eval_path}")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
