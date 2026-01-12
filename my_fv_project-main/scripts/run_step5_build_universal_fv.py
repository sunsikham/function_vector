#!/usr/bin/env python3
"""STEP 5: Build mean activations and a universal FV from hook captures."""

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.fv import build_fv, parse_heads
from fv.hooks import get_c_proj_pre_hook, reshape_resid_to_heads
from fv.io import prepare_run_dirs, resolve_out_dir, save_step5_artifacts, step5_paths
from fv.mean_acts import extract_slot_activation
from fv.model_config import get_model_config
from fv.prompting import build_two_shot_prompt
from fv.slots import compute_query_predictive_slot


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP 5 universal FV builder.")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument("--layer", type=int, default=0, help="Layer index (default: 0)")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of prompts (default: 20)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--slot",
        default="query_predictive",
        help="Slot selection (default: query_predictive)",
    )
    parser.add_argument(
        "--heads",
        default="0:0",
        help="Universal heads, e.g. '0:0,0:1,0:2'",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier (default: auto timestamp)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/<run_id>/artifacts/step5/)",
    )
    args = parser.parse_args()

    run_info = None
    if args.out_dir:
        out_dir = resolve_out_dir(args.out_dir)
    else:
        run_info = prepare_run_dirs(args.run_id)
        out_dir = os.path.join(run_info["artifacts_dir"], "step5")
    os.makedirs(out_dir, exist_ok=True)

    if args.slot != "query_predictive":
        print("Only slot 'query_predictive' is supported in this script.")
        return 1

    try:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import required libraries: {exc}")
        return 1

    model_cfg = get_model_config(args.model)
    if model_cfg is None:
        print(f"No model config found for '{args.model}'")
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        print(f"device: cuda ({gpu_name})")
    else:
        print("device: cpu")

    print(f"torch: {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"seed: {args.seed}")
    print(f"model: {args.model}")
    print(f"layer: {args.layer}")
    print(f"slot: {args.slot}")
    if run_info:
        print(f"run_id: {run_info['run_id']}")
    print(f"out_dir: {out_dir}")
    hook_target = model_cfg["hook_target"].format(layer=args.layer)
    reshape_rule = model_cfg["reshape"]
    print(
        "config: "
        f"n_heads={model_cfg['n_heads']} "
        f"head_dim={model_cfg['head_dim']} "
        f"resid_dim={model_cfg['resid_dim']} "
        f"hook_target={hook_target} "
        f"reshape={reshape_rule}"
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model)
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load model '{args.model}': {exc}")
        return 1

    model.to(device)
    model.eval()

    hidden_size = getattr(model.config, "n_embd", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "hidden_size", None)
    num_heads = getattr(model.config, "n_head", None)
    if num_heads is None:
        num_heads = getattr(model.config, "num_attention_heads", None)
    if hidden_size is None or num_heads is None:
        print("Missing model config for hidden_size or num_heads.")
        return 1

    if hidden_size % num_heads != 0:
        print("hidden_size must be divisible by n_heads.")
        return 1

    head_dim = hidden_size // num_heads
    resid_dim = num_heads * head_dim
    if num_heads != model_cfg["n_heads"] or head_dim != model_cfg["head_dim"]:
        print("Model config does not match expected head shape.")
        return 1
    if resid_dim != model_cfg["resid_dim"]:
        print("Model config does not match expected resid_dim.")
        return 1

    try:
        target_module, target_name = get_c_proj_pre_hook(model, args.layer)
    except ValueError as exc:
        print(str(exc))
        return 1

    target_name = hook_target

    hook_state = {"current": None, "printed": False, "errors": [], "slot_index": None}

    def pre_hook(_module, inputs):
        tensor = inputs[0] if inputs else None
        try:
            reshaped = reshape_resid_to_heads(tensor, num_heads, head_dim, resid_dim)
        except ValueError as exc:
            hook_state["errors"].append(str(exc))
            return

        slot_index = hook_state["slot_index"]
        if slot_index is None:
            hook_state["errors"].append("slot_index not set")
            return

        try:
            captured, _seq_len = extract_slot_activation(reshaped, slot_index)
        except ValueError as exc:
            hook_state["errors"].append(str(exc))
            return

        hook_state["current"] = captured.detach()

        if not hook_state["printed"]:
            print("hook called")
            print(
                "captured: "
                f"module={target_name} "
                f"shape={tuple(reshaped.shape)} "
                f"dtype={reshaped.dtype} "
                f"device={reshaped.device} "
                f"n_heads={num_heads} "
                f"head_dim={head_dim} "
                f"resid_dim={resid_dim}"
            )
            print(f"captured activation at slot_index shape: {tuple(captured.shape)}")
            hook_state["printed"] = True

    handle = target_module.register_forward_pre_hook(pre_hook)

    rng = random.Random(args.seed)
    sum_activations = torch.zeros((num_heads, head_dim), device=device)
    count = 0

    print(f"n_trials start: {args.n_trials}")
    slot_name = args.slot
    for trial_idx in range(args.n_trials):
        hook_state["current"] = None
        hook_state["printed"] = False
        hook_state["errors"].clear()

        prefix_str, full_str, _answer = build_two_shot_prompt(rng)
        try:
            slot_info = compute_query_predictive_slot(prefix_str, full_str, tokenizer)
        except ValueError as exc:
            print(str(exc))
            return 1

        s = slot_info["s"]
        slot_index = slot_info["slot_index"]
        seq_len = slot_info["seq_len"]
        target_id = slot_info["target_id"]
        target_token = slot_info["target_token"]

        print(
            "slot_debug: "
            f"trial={trial_idx} "
            f"slot_name={slot_name} "
            f"s={s} "
            f"slot_index={slot_index} "
            f"seq_len={seq_len} "
            f"target_id={target_id} "
            f"target_token={target_token}"
        )

        hook_state["slot_index"] = slot_index

        inputs = tokenizer(full_str, return_tensors="pt", add_special_tokens=False)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            _ = model(**inputs)

        if hook_state["errors"]:
            print("Hook error: " + "; ".join(hook_state["errors"]))
            handle.remove()
            return 1

        if hook_state["current"] is None:
            print("Hook did not capture a tensor.")
            handle.remove()
            return 1

        if hook_state["current"].shape != (num_heads, head_dim):
            print(
                f"Captured activation has wrong shape: {tuple(hook_state['current'].shape)}"
            )
            handle.remove()
            return 1

        sum_activations += hook_state["current"]
        count += 1

    handle.remove()
    print(f"n_trials done: {count}")

    if count == 0:
        print("No trials completed.")
        return 1

    mean_activations = sum_activations / count
    mean_norm = mean_activations.norm().item()
    mean_mean = mean_activations.mean().item()
    mean_std = mean_activations.std(unbiased=False).item()
    print(f"mean_activations shape: {tuple(mean_activations.shape)}")
    print(f"mean_activations stats: norm={mean_norm:.6f} mean={mean_mean:.6f} std={mean_std:.6f}")

    try:
        head_specs = parse_heads(args.heads)
    except ValueError as exc:
        print(str(exc))
        return 1

    if not head_specs:
        print("No heads provided. Use --heads like '0:0,0:1,0:2'.")
        return 1

    selected_heads = []
    for layer, head in head_specs:
        if layer != args.layer:
            print(f"Head list contains layer {layer}, expected {args.layer}.")
            return 1
        if head < 0 or head >= num_heads:
            print(f"Head index out of range: {head}")
            return 1
        selected_heads.append(head)

    fv = build_fv(mean_activations, selected_heads, head_dim, resid_dim)

    fv_norm = fv.norm().item()
    print(f"fv shape: {tuple(fv.shape)}")
    print(f"fv norm: {fv_norm:.6f}")
    assert fv.shape == (resid_dim,), "FV shape mismatch"
    assert fv_norm > 0, "FV norm is zero"

    paths = step5_paths(out_dir, args.model, args.layer, args.n_trials)

    metadata = {
        "model": args.model,
        "layer": args.layer,
        "hook_target": target_name,
        "slot": args.slot,
        "slot_name": slot_name,
        "slot_rule": "prefix_str -> s -> slot_index=s-1",
        "slot_index_rule": "s=len(tokenize(prefix_str)); slot_index=s-1; target_id=input_ids[s]",
        "heads": args.heads,
        "seed": args.seed,
        "n_trials": args.n_trials,
        "config": {
            "n_heads": num_heads,
            "head_dim": head_dim,
            "resid_dim": resid_dim,
            "hook_type": "pre",
            "reshape": reshape_rule,
        },
        "mean_activations": {
            "shape": list(mean_activations.shape),
            "norm": mean_norm,
        },
        "fv": {"shape": list(fv.shape), "norm": fv_norm},
        "paths": {"mean_activations": paths["mean_path"], "fv": paths["fv_path"]},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    saved_paths = save_step5_artifacts(
        out_dir,
        args.model,
        args.layer,
        args.n_trials,
        mean_activations,
        fv,
        metadata,
    )

    print(f"saved mean_activations: {saved_paths['mean_path']}")
    print(f"saved fv: {saved_paths['fv_path']}")
    print(f"metadata saved: {saved_paths['metadata_path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
