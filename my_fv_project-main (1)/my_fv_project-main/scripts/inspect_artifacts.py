#!/usr/bin/env python3
"""Inspect saved artifacts and metadata."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.io import resolve_run_dir

def load_metadata(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_metadata_in_dir(directory: str) -> Optional[str]:
    candidates = [
        name
        for name in os.listdir(directory)
        if name.startswith("metadata_") and name.endswith(".json")
    ]
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"Multiple metadata files found: {candidates}")
    return os.path.join(directory, candidates[0])


def infer_metadata_path_from_pt(pt_path: str) -> Optional[str]:
    base = os.path.basename(pt_path)
    directory = os.path.dirname(pt_path)
    if base.startswith("fv_") and base.endswith(".pt"):
        suffix = base[len("fv_") : -3]
        return os.path.join(directory, f"metadata_{suffix}.json")
    if base.startswith("mean_activations_") and base.endswith(".pt"):
        suffix = base[len("mean_activations_") : -3]
        return os.path.join(directory, f"metadata_{suffix}.json")
    return None


def gather_pt_paths(metadata: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    paths = metadata.get("paths") if isinstance(metadata.get("paths"), dict) else {}
    fv_path = paths.get("fv")
    mean_path = paths.get("mean_activations")
    return fv_path, mean_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect artifact tensors and metadata.")
    parser.add_argument(
        "--run_id",
        default=None,
        help="Run identifier to search under runs/<run_id>/artifacts",
    )
    parser.add_argument("path", help="Path to artifact (.pt), metadata.json, or directory")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime import check
        print(f"Failed to import torch: {exc}")
        return 1

    input_path = args.path
    if args.run_id and not os.path.isabs(input_path) and not os.path.exists(input_path):
        _, run_dir = resolve_run_dir(args.run_id)
        run_artifacts_dir = os.path.join(run_dir, "artifacts")
        candidate = os.path.join(run_artifacts_dir, input_path)
        if os.path.exists(candidate):
            input_path = candidate
    metadata_path = None
    metadata = None
    fv_path = None
    mean_path = None

    if os.path.isdir(input_path):
        try:
            metadata_path = find_metadata_in_dir(input_path)
        except ValueError as exc:
            print(str(exc))
            return 1
        if metadata_path is None:
            print("No metadata.json found in directory.")
            return 1
        metadata = load_metadata(metadata_path)
        fv_path, mean_path = gather_pt_paths(metadata)
    elif input_path.endswith(".json"):
        metadata_path = input_path
        metadata = load_metadata(metadata_path)
        fv_path, mean_path = gather_pt_paths(metadata)
    elif input_path.endswith(".pt"):
        base = os.path.basename(input_path)
        if base.startswith("mean_activations_"):
            mean_path = input_path
        else:
            fv_path = input_path
        metadata_path = infer_metadata_path_from_pt(input_path)
        if metadata_path and os.path.exists(metadata_path):
            metadata = load_metadata(metadata_path)
            stored_fv, stored_mean = gather_pt_paths(metadata)
            fv_path = stored_fv or fv_path
            mean_path = stored_mean
        else:
            metadata_path = None
    else:
        print("Unsupported path type. Provide a .pt, .json, or directory.")
        return 1

    if metadata_path:
        print(f"metadata: {metadata_path}")
    if metadata:
        model_name = metadata.get("model")
        if model_name:
            print(f"Config loaded for {model_name}")
        print(f"model: {metadata.get('model')}")
        print(f"layer: {metadata.get('layer')}")
        print(f"hook_target: {metadata.get('hook_target')}")
        print(f"slot: {metadata.get('slot')}")
        print(f"heads: {metadata.get('heads')}")
        print(f"seed: {metadata.get('seed')}")
        print(f"n_trials: {metadata.get('n_trials')}")
        if metadata.get("n_eval") is not None:
            print(f"n_eval: {metadata.get('n_eval')}")
        config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
        if config:
            print(
                "config: "
                f"n_heads={config.get('n_heads')} "
                f"head_dim={config.get('head_dim')} "
                f"resid_dim={config.get('resid_dim')} "
                f"hook_type={config.get('hook_type')}"
            )

    if fv_path:
        try:
            fv = torch.load(fv_path, map_location="cpu")
        except Exception as exc:
            print(f"Failed to load fv tensor: {exc}")
            return 1
        fv_norm = fv.norm().item() if hasattr(fv, "norm") else None
        print(f"fv_path: {fv_path}")
        print(f"fv shape: {tuple(fv.shape)}")
        if fv_norm is not None:
            print(f"fv norm: {fv_norm:.6f}")

        resid_dim = None
        if metadata and isinstance(metadata.get("config"), dict):
            resid_dim = metadata["config"].get("resid_dim")
        if resid_dim is not None:
            assert fv.shape == (resid_dim,), "FV shape does not match resid_dim"
            print("fv shape matches resid_dim")

    if mean_path:
        try:
            mean_act = torch.load(mean_path, map_location="cpu")
        except Exception as exc:
            print(f"Failed to load mean_activations tensor: {exc}")
            return 1
        mean_norm = mean_act.norm().item() if hasattr(mean_act, "norm") else None
        print(f"mean_activations path: {mean_path}")
        print(f"mean_activations shape: {tuple(mean_act.shape)}")
        if mean_norm is not None:
            print(f"mean_activations norm: {mean_norm:.6f}")

    if not metadata:
        print("metadata not loaded; cannot validate config")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
