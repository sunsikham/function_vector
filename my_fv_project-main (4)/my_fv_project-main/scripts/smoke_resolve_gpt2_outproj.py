#!/usr/bin/env python3
"""Smoke test for spec-driven GPT-2 out_proj resolution."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Spec-driven GPT-2 resolve smoke test.")
    parser.add_argument(
        "--negative",
        action="store_true",
        help="Run negative test with invalid blocks_path (expect failure).",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import transformers: {exc}")
        return 1

    from fv.adapters import resolve_blocks
    from fv.hooks import get_out_proj_pre_hook_target
    from fv.model_spec import get_model_spec

    spec = get_model_spec("gpt2")
    try:
        model = AutoModelForCausalLM.from_pretrained(spec.hf_model_id)
    except Exception as exc:  # pragma: no cover - runtime load check
        print(f"Failed to load model '{spec.hf_model_id}': {exc}")
        return 1

    if args.negative:
        bad_spec = replace(spec, blocks_path="transformer.BAD")
        try:
            _ = resolve_blocks(model, bad_spec)
        except Exception as exc:
            print(f"negative test error: {exc}")
            return 0
        print("negative test unexpectedly succeeded")
        return 1

    print(f"spec: {spec.name}")
    print(
        "spec dims: "
        f"hidden_size={spec.hidden_size} n_heads={spec.n_heads} head_dim={spec.head_dim}"
    )

    try:
        out_proj, path = get_out_proj_pre_hook_target(
            model, layer_idx=0, spec_name=spec.name
        )
    except Exception as exc:
        print(f"Failed to resolve out_proj (layer 0): {exc}")
        return 1
    print(f"layer 0 path: {path}")
    print(f"layer 0 module type: {type(out_proj).__name__}")

    try:
        _, path_11 = get_out_proj_pre_hook_target(
            model, layer_idx=11, spec_name=spec.name
        )
    except Exception as exc:
        print(f"Failed to resolve out_proj (layer 11): {exc}")
        return 1
    print(f"layer 11 path: {path_11}")

    try:
        _ = get_out_proj_pre_hook_target(model, layer_idx=12, spec_name=spec.name)
    except Exception as exc:
        print(f"layer 12 error: {exc}")
        return 0
    print("layer 12 unexpectedly succeeded")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
