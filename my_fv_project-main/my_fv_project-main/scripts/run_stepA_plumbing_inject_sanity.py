#!/usr/bin/env python3
"""Step A: plumbing sanity check for hook/slot/injection."""

import argparse
import os
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.hooks import get_c_proj_pre_hook, reshape_resid_to_heads
from fv.intervene import make_residual_injection_hook
from fv.io import prepare_run_dirs, resolve_out_dir, save_json
from fv.mean_acts import extract_slot_activation
from fv.model_config import get_model_config
from fv.prompting import build_zero_shot_prompt
from fv.slots import compute_query_predictive_slot


def make_logger(log_path: str):
    log_file = open(log_path, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    return log, log_file


def get_answer_first_id(answer: str, tokenizer) -> int:
    answer_with_space = " " + answer
    token_ids = tokenizer.encode(answer_with_space, add_special_tokens=False)
    if not token_ids:
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Failed to tokenize answer: '{answer}'")
    return token_ids[0]


def resolve_head_config(model):
    hidden_size = getattr(model.config, "n_embd", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "hidden_size", None)
    num_heads = getattr(model.config, "n_head", None)
    if num_heads is None:
        num_heads = getattr(model.config, "num_attention_heads", None)
    if hidden_size is None or num_heads is None:
        raise ValueError("Missing model config for hidden_size or num_heads.")
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads.")
    head_dim = hidden_size // num_heads
    resid_dim = num_heads * head_dim
    return num_heads, head_dim, resid_dim, hidden_size


def main() -> int:
    parser = argparse.ArgumentParser(description="Step A plumbing sanity check.")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default: 0)")
    parser.add_argument("--head", type=int, default=0, help="Head index (default: 0)")
    parser.add_argument("--n_trials", type=int, default=3, help="Number of trials (default: 3)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Injection scale (default: 1.0)")
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

    run_info = prepare_run_dirs(args.run_id)
    if args.out_dir:
        artifacts_dir = resolve_out_dir(args.out_dir)
    else:
        artifacts_dir = run_info["artifacts_dir"]
    os.makedirs(artifacts_dir, exist_ok=True)

    log_path = os.path.join(run_info["logs_dir"], "stepA_plumbing.log")
    log, log_file = make_logger(log_path)

    log("stepA plumbing sanity start")
    log(f"run_id: {run_info['run_id']}")
    log(f"artifacts_dir: {artifacts_dir}")
    log(f"log_path: {log_path}")
    log(f"model: {args.model}")
    log(f"layer: {args.layer}")
    log(f"head: {args.head}")
    log(f"n_trials: {args.n_trials}")
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
        num_heads, head_dim, resid_dim, hidden_size = resolve_head_config(model)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    if model_cfg:
        mismatch = []
        if model_cfg.get("n_heads") != num_heads:
            mismatch.append("n_heads")
        if model_cfg.get("head_dim") != head_dim:
            mismatch.append("head_dim")
        if model_cfg.get("resid_dim") != resid_dim:
            mismatch.append("resid_dim")
        if mismatch:
            log("warning: model_config mismatch for " + ",".join(mismatch))

    try:
        target_module, target_name = get_c_proj_pre_hook(model, args.layer)
    except ValueError as exc:
        log(str(exc))
        log_file.close()
        return 1

    if hidden_size != resid_dim:
        log("hidden_size does not match resid_dim")
        log_file.close()
        return 1

    if args.head < 0 or args.head >= num_heads:
        log("head index out of range")
        log_file.close()
        return 1

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        log("Model does not expose transformer.h blocks.")
        log_file.close()
        return 1
    if args.layer < 0 or args.layer >= len(model.transformer.h):
        log("Layer index out of range.")
        log_file.close()
        return 1
    block = model.transformer.h[args.layer]

    injection_state = {
        "fv": None,
        "alpha": args.alpha,
        "batch_indices": None,
        "last_indices": None,
    }
    inject_hook = make_residual_injection_hook(injection_state)

    rng = random.Random(0)
    trials = []
    delta_values = []

    for trial_idx in range(args.n_trials):
        prefix_str, answer_str = build_zero_shot_prompt(rng)
        full_str = f"{prefix_str} {answer_str}"

        try:
            slot_info = compute_query_predictive_slot(prefix_str, full_str, tokenizer)
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1

        s = slot_info["s"]
        slot_index = slot_info["slot_index"]
        seq_len = slot_info["seq_len"]
        target_id = slot_info["target_id"]
        target_token = slot_info["target_token"]
        input_ids = slot_info["input_ids"]
        slot_token = tokenizer.convert_ids_to_tokens(input_ids[slot_index])

        try:
            answer_first_id = get_answer_first_id(answer_str, tokenizer)
        except ValueError as exc:
            log(str(exc))
            log_file.close()
            return 1

        if answer_first_id != target_id:
            log(
                "answer first token mismatch: "
                f"answer_first_id={answer_first_id} target_id={target_id}"
            )
            log_file.close()
            return 1

        log(
            "slot_debug: "
            f"trial={trial_idx} "
            f"s={s} "
            f"slot_index={slot_index} "
            f"slot_token={slot_token} "
            f"seq_len={seq_len} "
            f"target_id={target_id} "
            f"target_token={target_token}"
        )

        hook_state = {"slot_index": slot_index, "captured": None, "errors": [], "shape": None}

        def pre_hook(_module, inputs):
            tensor = inputs[0] if inputs else None
            try:
                reshaped = reshape_resid_to_heads(tensor, num_heads, head_dim, resid_dim)
            except ValueError as exc:
                hook_state["errors"].append(str(exc))
                return
            hook_state["shape"] = tuple(reshaped.shape)
            try:
                captured, _seq_len = extract_slot_activation(reshaped, slot_index)
            except ValueError as exc:
                hook_state["errors"].append(str(exc))
                return
            hook_state["captured"] = captured.detach()

        handle = target_module.register_forward_pre_hook(pre_hook)
        inputs_full = tokenizer(full_str, return_tensors="pt", add_special_tokens=False)
        inputs_full = {key: value.to(device) for key, value in inputs_full.items()}
        with torch.inference_mode():
            _ = model(**inputs_full)
        handle.remove()

        if hook_state["errors"]:
            log("hook error: " + "; ".join(hook_state["errors"]))
            log_file.close()
            return 1
        if hook_state["captured"] is None:
            log("hook did not capture a tensor")
            log_file.close()
            return 1

        captured = hook_state["captured"]
        captured_shape = tuple(captured.shape)
        reshape_shape = hook_state["shape"]

        log(
            "capture_debug: "
            f"trial={trial_idx} "
            f"module={target_name} "
            f"reshape_shape={reshape_shape} "
            f"captured_shape={captured_shape} "
            f"n_heads={num_heads} "
            f"head_dim={head_dim} "
            f"resid_dim={resid_dim}"
        )

        head_vector = captured[args.head]
        head_only_fv = torch.zeros((resid_dim,), device=device, dtype=head_vector.dtype)
        start = args.head * head_dim
        end = start + head_dim
        head_only_fv[start:end] = head_vector

        inputs = tokenizer(prefix_str, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            last_indices = attention_mask.sum(dim=1) - 1
        else:
            last_indices = torch.tensor(
                [inputs["input_ids"].shape[1] - 1], device=device
            )
        batch_indices = torch.arange(inputs["input_ids"].shape[0], device=device)
        target_ids_tensor = torch.tensor([target_id], device=device)

        with torch.inference_mode():
            outputs = model(**inputs)
        logits = outputs.logits
        last_logits = logits[batch_indices, last_indices]
        baseline_logit = last_logits[batch_indices, target_ids_tensor].item()

        injection_state["fv"] = head_only_fv
        injection_state["batch_indices"] = batch_indices
        injection_state["last_indices"] = last_indices
        handle = block.register_forward_hook(inject_hook)
        with torch.inference_mode():
            outputs_injected = model(**inputs)
        handle.remove()

        logits_injected = outputs_injected.logits
        last_logits_injected = logits_injected[batch_indices, last_indices]
        injected_logit = last_logits_injected[batch_indices, target_ids_tensor].item()
        delta_logit = injected_logit - baseline_logit
        delta_values.append(delta_logit)

        log(
            "delta_debug: "
            f"trial={trial_idx} "
            f"baseline_logit={baseline_logit:.6f} "
            f"injected_logit={injected_logit:.6f} "
            f"delta_logit={delta_logit:.6f}"
        )

        trials.append(
            {
                "trial_idx": trial_idx,
                "prefix_str": prefix_str,
                "answer_str": answer_str,
                "seq_len": seq_len,
                "s": s,
                "slot_index": slot_index,
                "slot_token": slot_token,
                "target_id": target_id,
                "target_token": target_token,
                "captured_shape": list(captured_shape),
                "head": args.head,
                "head_dim": head_dim,
                "resid_dim": resid_dim,
                "baseline_logit": baseline_logit,
                "injected_logit": injected_logit,
                "delta_logit": delta_logit,
            }
        )

    mean_delta = sum(delta_values) / len(delta_values) if delta_values else 0.0
    std_delta = statistics.pstdev(delta_values) if len(delta_values) > 1 else 0.0
    any_nonzero = any(abs(value) > 1e-6 for value in delta_values)

    summary = {
        "n_trials": args.n_trials,
        "mean_delta_logit": mean_delta,
        "std_delta_logit": std_delta,
        "any_nonzero_delta": any_nonzero,
    }

    payload = {"trials": trials, "summary": summary}

    out_path = os.path.join(artifacts_dir, "stepA_plumbing_results.json")
    save_json(out_path, payload)

    log(f"summary: mean_delta_logit={mean_delta:.6f} std_delta_logit={std_delta:.6f}")
    log(f"any_nonzero_delta: {any_nonzero}")
    log(f"saved results: {out_path}")
    log_file.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
