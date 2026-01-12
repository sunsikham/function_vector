#!/usr/bin/env python3
"""STEP 6: Evaluate FV injection on 0-shot antonym prompts."""

import argparse
import os
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.dataset_loader import load_pairs_antonym
from fv.hf_loader import load_hf_model_and_tokenizer
from fv.model_spec import get_model_spec
from fv.intervene import make_residual_injection_hook
from fv.io import load_json, prepare_run_dirs, resolve_out_dir, save_step6_results
from fv.adapters import infer_head_dims, resolve_blocks
from fv.slots import compute_query_predictive_slot


def summarize_prompt(prompt: str, limit: int = 80) -> str:
    if len(prompt) <= limit:
        return prompt
    return prompt[: limit - 3] + "..."


def infer_leading_space(answer: str, tokenizer) -> bool:
    with_space = " " + answer
    ids_with_space = tokenizer.encode(with_space, add_special_tokens=False)
    if ids_with_space:
        return True
    token_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Failed to tokenize answer: '{answer}'")
    return False


def parse_answer_override(value: str, tokenizer) -> Dict[str, int]:
    if value == "auto":
        return {}
    if value.isdigit():
        token_id = int(value)
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        return {"override_id": token_id, "override_str": token_str}
    token_ids = tokenizer.encode(value, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Failed to tokenize --answer_first_token: '{value}'")
    token_id = token_ids[0]
    token_str = tokenizer.convert_ids_to_tokens(token_id)
    return {"override_id": token_id, "override_str": token_str}


def resolve_device(device_str: str, torch_module):
    if device_str == "auto":
        return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    if device_str == "cuda":
        if not torch_module.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch_module.device("cuda")
    if device_str == "cpu":
        return torch_module.device("cpu")
    raise ValueError(f"Unknown device option: {device_str}")


def resolve_dtype(dtype_str: str, torch_module):
    mapping = {
        "fp32": torch_module.float32,
        "fp16": torch_module.float16,
        "bf16": torch_module.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype option: {dtype_str}")
    return mapping[dtype_str]


def check_llama_target_paths(model, spec, logger):
    def resolve_required(root, dotted_path: str, label: str):
        current = root
        parts = dotted_path.split(".")
        for idx, part in enumerate(parts):
            if not hasattr(current, part):
                prefix = ".".join(parts[: idx + 1])
                raise ValueError(
                    f"Spec '{spec.name}' failed to resolve {label} '{dotted_path}': "
                    f"missing '{part}' at '{prefix}'"
                )
            current = getattr(current, part)
        return current

    blocks = resolve_required(model, spec.blocks_path, label="blocks_path")
    logger(f"llama target: blocks_path={spec.blocks_path} exists=True")
    blocks_list = list(blocks)
    if not blocks_list:
        raise ValueError(
            f"Spec '{spec.name}' failed to resolve blocks_path '{spec.blocks_path}': "
            "no blocks found"
        )
    block0 = blocks_list[0]
    attn = resolve_required(block0, spec.attn_path_in_block, label="attn_path_in_block")
    logger(
        f"llama target: attn_path_in_block={spec.attn_path_in_block} exists=True"
    )
    resolve_required(attn, spec.out_proj_path_in_attn, label="out_proj_path_in_attn")
    logger(
        f"llama target: out_proj_path_in_attn={spec.out_proj_path_in_attn} exists=True"
    )

    mlp = resolve_required(block0, "mlp", label="mlp")
    logger("llama target: mlp exists=True")
    for proj_name in ("down_proj", "up_proj", "gate_proj"):
        resolve_required(mlp, proj_name, label=f"mlp.{proj_name}")
        logger(f"llama target: mlp.{proj_name} exists=True")


def compute_slot_with_fallback(prefix_str: str, full_str: str, tokenizer, log):
    try:
        return compute_query_predictive_slot(prefix_str, full_str, tokenizer), False
    except ValueError as exc:
        message = str(exc)
        if "Target id mismatch" not in message:
            raise
        trimmed_prefix = prefix_str.rstrip(" ")
        if trimmed_prefix == prefix_str:
            raise
        log("retrying slot computation with trimmed prefix space")
        return compute_query_predictive_slot(trimmed_prefix, full_str, tokenizer), True


def compute_token_scores(logits, target_ids):
    import torch

    logits32 = logits.float()
    if torch.is_tensor(target_ids):
        gathered = logits32.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    else:
        gathered = logits32[..., target_ids]
    logprob = gathered - torch.logsumexp(logits32, dim=-1)
    p = torch.exp(logprob)
    return {"logit": gathered, "logprob": logprob, "p": p}


def std(values):
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values)


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 6 FV injection eval.")
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
        "--edit_layer",
        type=int,
        default=0,
        help="Layer index to edit (default: 0)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Deprecated alias for --edit_layer",
    )
    parser.add_argument(
        "--run_id_stepE",
        default=None,
        help="StepE run_id for fv_global_resid artifacts",
    )
    parser.add_argument(
        "--fv_global_path",
        default=None,
        help="Path to fv_global_resid.pt (overrides --run_id_stepE)",
    )
    parser.add_argument(
        "--fv_global_meta_path",
        default=None,
        help="Path to fv_global_resid_meta.json (optional override)",
    )
    parser.add_argument(
        "--use_fv_by_layer",
        action="store_true",
        help="Use fv_by_layer.pt for per-layer injection (default: off)",
    )
    parser.add_argument(
        "--fv_by_layer_path",
        default=None,
        help="Path to fv_by_layer.pt (requires --use_fv_by_layer)",
    )
    parser.add_argument(
        "--fv_path",
        default=None,
        help="Deprecated alias for --fv_global_path",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Injection scale (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--n_eval", type=int, default=20, help="Number of eval prompts")
    parser.add_argument(
        "--answer_first_token",
        default="auto",
        help="Override target token (string or id), or 'auto'",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cpu", "cuda"],
        help="Device selection (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["fp32", "fp16", "bf16"],
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
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for eval prompts (default: 1)",
    )
    parser.add_argument(
        "--score_key",
        default="mean_delta_logprob",
        choices=["delta_acc", "mean_delta_logprob", "mean_delta_p", "mean_delta_logit"],
        help="Score key for downstream selection (default: mean_delta_logprob)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/<run_id>/artifacts/step6/)",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier (default: auto timestamp)",
    )
    args = parser.parse_args()

    if args.layer is not None:
        args.edit_layer = args.layer
    if args.fv_path and not args.fv_global_path:
        args.fv_global_path = args.fv_path

    run_info = None
    if args.out_dir:
        out_dir = resolve_out_dir(args.out_dir)
    else:
        run_info = prepare_run_dirs(args.run_id)
        out_dir = os.path.join(run_info["artifacts_dir"], "step6")
    os.makedirs(out_dir, exist_ok=True)
    if run_info:
        log_path = os.path.join(run_info["logs_dir"], "step6.log")
    else:
        log_path = os.path.join(out_dir, "step6.log")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    log("step6 resid start")

    try:
        import torch
        import transformers
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import required libraries: {exc}")
        return 1

    try:
        spec = get_model_spec(args.model_spec)
    except ValueError as exc:
        print(str(exc))
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device in (None, "auto"):
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype is None:
        args.dtype = "fp16" if args.device == "cuda" else "fp32"

    try:
        device = resolve_device(args.device, torch)
    except ValueError as exc:
        print(str(exc))
        return 1

    if device.type == "cpu" and args.dtype in {"fp16", "bf16"}:
        print(f"cpu does not support {args.dtype}; forcing fp32")
        args.dtype = "fp32"

    try:
        dtype = resolve_dtype(args.dtype, torch)
    except ValueError as exc:
        print(str(exc))
        return 1

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"device: cuda ({gpu_name})")
    else:
        print("device: cpu")

    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"seed: {args.seed}")
    print(f"model: {args.model}")
    print(f"dataset_path: {args.dataset_path}")
    print(f"model_spec: {args.model_spec}")
    print(f"edit_layer: {args.edit_layer}")
    print(f"use_fv_by_layer: {args.use_fv_by_layer}")
    if args.run_id_stepE:
        print(f"run_id_stepE: {args.run_id_stepE}")
    print(f"alpha: {args.alpha}")
    print(f"dtype: {args.dtype}")
    print(f"quant: {args.quant}")
    print(f"score_key: {args.score_key}")
    if args.device_map:
        print(f"device_map: {args.device_map}")
    print(f"batch_size: {args.batch_size}")
    if run_info:
        print(f"run_id: {run_info['run_id']}")
    print(f"out_dir: {out_dir}")

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
        print(
            "hf_loader diagnostics: "
            + " ".join(
                f"{key}={value}" for key, value in diagnostics.items()
            )
        )
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load model '{args.model}': {exc}")
        return 1

    resolved_quant = diagnostics.get("resolved_quant") if diagnostics else None
    if args.device_map:
        print("device_map provided; skipping model.to(device)")
        try:
            device = next(model.parameters()).device
        except StopIteration:
            pass
        print(f"input device: {device}")
    elif resolved_quant in {"4bit", "8bit"}:
        print("quantized model; skipping model.to(device)")
    else:
        model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if spec.name == "llama3":
        try:
            check_llama_target_paths(model, spec, logger=print)
        except ValueError as exc:
            print(str(exc))
            return 1
    try:
        pairs = load_pairs_antonym(args.dataset_path, canonical_by_input=True)
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        return 1
    if not pairs:
        print("Dataset returned no valid pairs.")
        return 1
    example_x, example_y = random.Random(args.seed).choice(pairs)
    example_prefix = f"Q: {example_x}\nA: "
    print(f"example_pair: input='{example_x}' output='{example_y}'")
    print(f"prefix_endswith_A_space: {example_prefix.endswith('A: ')}")
    try:
        dims = infer_head_dims(model, spec_name=args.model_spec)
    except ValueError as exc:
        print(str(exc))
        return 1

    n_heads = int(dims["n_heads"])
    head_dim = int(dims["head_dim"])
    resid_dim = int(dims["hidden_size"])
    reshape_rule = "resid_to_heads"
    hook_target = (
        f"{spec.blocks_path}.{{layer}}."
        f"{spec.attn_path_in_block}.{spec.out_proj_path_in_attn}"
    )
    print(
        "config: "
        f"n_heads={n_heads} "
        f"head_dim={head_dim} "
        f"resid_dim={resid_dim} "
        f"reshape={reshape_rule} "
        f"hook_target={hook_target}"
    )

    try:
        blocks = resolve_blocks(model, spec, logger=log)
    except ValueError as exc:
        print(str(exc))
        return 1

    if args.edit_layer < 0 or args.edit_layer >= len(blocks):
        print(
            "edit_layer out of range: "
            f"{args.edit_layer} not in [0, {len(blocks) - 1}]"
        )
        return 1

    artifact_base = None
    if args.run_id_stepE:
        artifact_base = resolve_out_dir(os.path.join("runs", args.run_id_stepE, "artifacts"))

    fv_source = "fv_global_resid"
    token_rule = "unknown"
    fv_meta = None
    if args.use_fv_by_layer:
        if args.fv_by_layer_path:
            fv_by_layer_path = resolve_out_dir(args.fv_by_layer_path)
        elif artifact_base:
            fv_by_layer_path = os.path.join(artifact_base, "fv_by_layer.pt")
        else:
            print("fv_by_layer requires --run_id_stepE or --fv_by_layer_path")
            return 1
        try:
            fv_blob = torch.load(fv_by_layer_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - runtime load check
            print(f"Failed to load fv_by_layer from '{fv_by_layer_path}': {exc}")
            return 1
        fv_by_layer = fv_blob.get("fv_by_layer", fv_blob)
        if args.edit_layer not in fv_by_layer and str(args.edit_layer) in fv_by_layer:
            fv_by_layer = {int(k): v for k, v in fv_by_layer.items()}
        if args.edit_layer not in fv_by_layer:
            print(f"edit_layer {args.edit_layer} missing in fv_by_layer")
            return 1
        fv = fv_by_layer[args.edit_layer]
        fv_source = "fv_by_layer"
    else:
        if args.fv_global_path:
            fv_global_path = resolve_out_dir(args.fv_global_path)
            if args.fv_global_meta_path:
                fv_global_meta_path = resolve_out_dir(args.fv_global_meta_path)
            else:
                fv_global_meta_path = os.path.join(
                    os.path.dirname(fv_global_path), "fv_global_resid_meta.json"
                )
        elif artifact_base:
            fv_global_path = os.path.join(artifact_base, "fv_global_resid.pt")
            fv_global_meta_path = os.path.join(artifact_base, "fv_global_resid_meta.json")
        else:
            print("fv_global_resid requires --run_id_stepE or --fv_global_path")
            return 1
        try:
            fv_blob = torch.load(fv_global_path, map_location="cpu")
        except Exception as exc:  # pragma: no cover - runtime load check
            print(f"Failed to load fv_global_resid from '{fv_global_path}': {exc}")
            return 1
        fv = fv_blob.get("fv_global_resid", fv_blob)
        if not os.path.exists(fv_global_meta_path):
            print(f"Missing fv_global_resid metadata: {fv_global_meta_path}")
            return 1
        try:
            meta = load_json(fv_global_meta_path)
        except Exception as exc:
            print(f"Failed to load fv_global_resid metadata: {exc}")
            return 1
        fv_meta = meta
        token_rule = meta.get("token_position_rule", "unknown")

    if fv.dim() != 1:
        print(f"FV must be 1D, got shape {tuple(fv.shape)}")
        return 1

    if fv.shape[0] != resid_dim:
        print(f"FV size {fv.shape[0]} does not match hidden_size {resid_dim}")
        return 1

    fv = fv.to(device=device, dtype=dtype)
    fv_norm = fv.norm().item()
    print(f"fv shape: {tuple(fv.shape)}")
    print(f"fv norm: {fv_norm:.6f}")

    print(f"fv_source: {fv_source}")
    print(f"token_rule: {token_rule}")
    log(f"edit_layer: {args.edit_layer}")
    log(f"alpha: {args.alpha}")
    log(f"token_rule: {token_rule}")
    log(f"fv_norm: {fv_norm:.6f}")
    log(f"score_key: {args.score_key}")

    injection_state = {
        "fv": fv,
        "alpha": args.alpha,
        "batch_indices": None,
        "last_indices": None,
    }
    inject_hook = make_residual_injection_hook(injection_state)
    target_block = blocks[args.edit_layer]

    if args.batch_size < 1:
        print("batch_size must be >= 1")
        return 1

    results = []
    sum_acc_base = 0.0
    sum_acc_with = 0.0
    sum_logit_base = 0.0
    sum_logit_with = 0.0
    sum_logprob_base = 0.0
    sum_logprob_with = 0.0
    sum_p_base = 0.0
    sum_p_with = 0.0
    sum_delta_logit = 0.0
    sum_delta_logprob = 0.0
    sum_delta_p = 0.0
    delta_logit_vals = []
    delta_logprob_vals = []
    delta_p_vals = []
    fallback_used_count = 0

    override_info = {}
    try:
        override_info = parse_answer_override(args.answer_first_token, tokenizer)
    except ValueError as exc:
        print(str(exc))
        return 1

    if override_info:
        print(
            "override target: "
            f"id={override_info['override_id']} "
            f"token={override_info['override_str']}"
        )

    print(f"n_eval start: {args.n_eval}")
    idx = 0
    while idx < args.n_eval:
        batch_prompts = []
        batch_answers = []
        batch_targets = []
        batch_target_tokens = []
        batch_leading_spaces = []
        batch_used_ids = []
        batch_used_tokens = []

        remaining = args.n_eval - idx
        batch_size = min(args.batch_size, remaining)
        for offset in range(batch_size):
            sample_idx = idx + offset
            sample_rng = random.Random(args.seed + sample_idx)
            x_val, y_val = sample_rng.choice(pairs)
            prefix_str = f"Q: {x_val}\nA: "
            full_str = prefix_str + y_val
            try:
                slot_info, fallback_used = compute_slot_with_fallback(
                    prefix_str, full_str, tokenizer, log
                )
            except ValueError as exc:
                print(str(exc))
                return 1
            if fallback_used:
                fallback_used_count += 1
            target_id = slot_info["target_id"]
            target_token = tokenizer.convert_ids_to_tokens(target_id)
            try:
                leading_space = infer_leading_space(y_val, tokenizer)
            except ValueError as exc:
                print(str(exc))
                return 1

            target_used = target_id
            target_used_token = target_token
            if override_info:
                target_used = override_info["override_id"]
                target_used_token = override_info["override_str"]

            batch_prompts.append(prefix_str)
            batch_answers.append(y_val)
            batch_targets.append(target_id)
            batch_target_tokens.append(target_token)
            batch_leading_spaces.append(leading_space)
            batch_used_ids.append(target_used)
            batch_used_tokens.append(target_used_token)

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            print("Missing attention_mask in tokenizer output.")
            return 1
        last_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(batch_size, device=device)
        target_ids_tensor = torch.tensor(batch_used_ids, device=device)

        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits
        last_logits = logits[batch_indices, last_indices]
        base_scores = compute_token_scores(last_logits, target_ids_tensor)
        pred_id_base = torch.argmax(last_logits, dim=-1)

        injection_state["last_indices"] = last_indices
        injection_state["batch_indices"] = batch_indices
        injection_state["calls"] = 0
        handle = target_block.register_forward_hook(inject_hook)
        with torch.inference_mode():
            outputs_fv = model(**inputs)
        handle.remove()
        log(f"residual hook calls (layer {args.edit_layer}): {injection_state['calls']}")

        logits_fv = outputs_fv.logits
        last_logits_fv = logits_fv[batch_indices, last_indices]
        with_scores = compute_token_scores(last_logits_fv, target_ids_tensor)
        pred_id_with = torch.argmax(last_logits_fv, dim=-1)

        for i in range(batch_size):
            base_prob = base_scores["p"][i].item()
            with_prob = with_scores["p"][i].item()
            base_logit = base_scores["logit"][i].item()
            with_logit = with_scores["logit"][i].item()
            base_logprob = base_scores["logprob"][i].item()
            with_logprob = with_scores["logprob"][i].item()
            delta_p = with_prob - base_prob
            delta_logit = with_logit - base_logit
            delta_logprob = with_logprob - base_logprob
            acc_base = 1.0 if pred_id_base[i].item() == batch_used_ids[i] else 0.0
            acc_with = 1.0 if pred_id_with[i].item() == batch_used_ids[i] else 0.0

            sum_acc_base += acc_base
            sum_acc_with += acc_with
            sum_logit_base += base_logit
            sum_logit_with += with_logit
            sum_logprob_base += base_logprob
            sum_logprob_with += with_logprob
            sum_p_base += base_prob
            sum_p_with += with_prob
            sum_delta_logit += delta_logit
            sum_delta_logprob += delta_logprob
            sum_delta_p += delta_p
            delta_logit_vals.append(delta_logit)
            delta_logprob_vals.append(delta_logprob)
            delta_p_vals.append(delta_p)

            prompt_summary = summarize_prompt(batch_prompts[i])
            print(
                "sample: "
                f"idx={idx} "
                f"prompt='{prompt_summary}' "
                f"answer='{batch_answers[i]}' "
                f"target_id={batch_targets[i]} "
                f"target_token={batch_target_tokens[i]} "
                f"leading_space={batch_leading_spaces[i]} "
                f"used_id={batch_used_ids[i]} "
                f"used_token={batch_used_tokens[i]} "
                f"p_base={base_prob:.6f} "
                f"p_with={with_prob:.6f} "
                f"delta={delta_p:.6f}"
            )

            results.append(
                {
                    "idx": idx,
                    "prompt": batch_prompts[i],
                    "answer": batch_answers[i],
                    "target_id": batch_targets[i],
                    "target_token": batch_target_tokens[i],
                    "leading_space": batch_leading_spaces[i],
                    "used_id": batch_used_ids[i],
                    "used_token": batch_used_tokens[i],
                    "pred_id_base": int(pred_id_base[i].item()),
                    "pred_id_with": int(pred_id_with[i].item()),
                    "accuracy_base": acc_base,
                    "accuracy_with": acc_with,
                    "delta_acc": acc_with - acc_base,
                    "logit_base": base_logit,
                    "logit_with": with_logit,
                    "delta_logit": delta_logit,
                    "logprob_base": base_logprob,
                    "logprob_with": with_logprob,
                    "delta_logprob": delta_logprob,
                    "p_base": base_prob,
                    "p_with": with_prob,
                    "delta_p": delta_p,
                }
            )
            idx += 1

    print(f"fallback_used_count: {fallback_used_count}")
    log(f"fallback_used_count: {fallback_used_count}")

    denom = args.n_eval if args.n_eval else 1
    acc_base = sum_acc_base / denom
    acc_with = sum_acc_with / denom
    delta_acc = acc_with - acc_base
    mean_logit_base = sum_logit_base / denom
    mean_logit_with = sum_logit_with / denom
    mean_logprob_base = sum_logprob_base / denom
    mean_logprob_with = sum_logprob_with / denom
    mean_p_base = sum_p_base / denom
    mean_p_with = sum_p_with / denom
    mean_delta_logit = sum_delta_logit / denom
    mean_delta_logprob = sum_delta_logprob / denom
    mean_delta_p = sum_delta_p / denom
    std_delta_logit = std(delta_logit_vals)
    std_delta_logprob = std(delta_logprob_vals)
    std_delta_p = std(delta_p_vals)

    print(f"acc_base: {acc_base:.6f}")
    print(f"acc_with: {acc_with:.6f}")
    print(f"delta_acc: {delta_acc:.6f}")
    print(f"mean(logit_base): {mean_logit_base:.6f}")
    print(f"mean(logit_with): {mean_logit_with:.6f}")
    print(f"mean(delta logit): {mean_delta_logit:.6f}")
    print(f"mean(logprob_base): {mean_logprob_base:.6f}")
    print(f"mean(logprob_with): {mean_logprob_with:.6f}")
    print(f"mean(delta logprob): {mean_delta_logprob:.6f}")
    print(f"mean(p_base): {mean_p_base:.6f}")
    print(f"mean(p_with): {mean_p_with:.6f}")
    print(f"mean(delta p): {mean_delta_p:.6f}")

    step6_metadata = {
        "model": args.model,
        "model_spec": args.model_spec,
        "edit_layer": args.edit_layer,
        "injection_point": "residual",
        "slot": "query_predictive",
        "slot_name": "query_predictive",
        "slot_index_rule": "s=len(tokenize(prefix_str)); slot_index=s-1; target_id=input_ids[s]",
        "token_rule": token_rule,
        "score_key": args.score_key,
        "heads": fv_meta.get("heads") if fv_meta else None,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "alpha": args.alpha,
        "fv_source": fv_source,
        "config": {
            "n_heads": n_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "hook_type": "pre",
            "reshape": reshape_rule,
            "hook_target": hook_target,
            "dtype": args.dtype,
            "device": str(device),
            "batch_size": args.batch_size,
        },
        "fv": {"shape": list(fv.shape), "norm": fv_norm},
        "fv_meta": fv_meta,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    payload = {
        "config": {
            "model": args.model,
            "model_spec": args.model_spec,
            "edit_layer": args.edit_layer,
            "fv_source": fv_source,
            "seed": args.seed,
            "n_eval": args.n_eval,
            "answer_first_token": args.answer_first_token,
            "alpha": args.alpha,
            "dtype": args.dtype,
            "device": str(device),
            "batch_size": args.batch_size,
            "score_key": args.score_key,
        },
        "fv": {"shape": list(fv.shape), "norm": fv_norm},
        "summary": {
            "acc_base": acc_base,
            "acc_with": acc_with,
            "delta_acc": delta_acc,
            "mean_logit_base": mean_logit_base,
            "mean_logit_with": mean_logit_with,
            "mean_delta_logit": mean_delta_logit,
            "std_delta_logit": std_delta_logit,
            "mean_logprob_base": mean_logprob_base,
            "mean_logprob_with": mean_logprob_with,
            "mean_delta_logprob": mean_delta_logprob,
            "std_delta_logprob": std_delta_logprob,
            "mean_p_base": mean_p_base,
            "mean_p_with": mean_p_with,
            "mean_delta_p": mean_delta_p,
            "std_delta_p": std_delta_p,
        },
        "metadata": step6_metadata,
        "results": results,
    }

    saved_paths = save_step6_results(
        out_dir,
        args.model,
        args.edit_layer,
        args.n_eval,
        payload,
        step6_metadata,
    )

    print(f"saved results: {saved_paths['results_path']}")
    log(f"saved results: {saved_paths['results_path']}")
    print(f"metadata saved: {saved_paths['metadata_path']}")
    log(f"metadata saved: {saved_paths['metadata_path']}")
    log_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
