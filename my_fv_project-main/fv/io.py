"""Artifact IO helpers."""

import csv
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def resolve_out_dir(out_dir: str) -> str:
    path = Path(out_dir)
    if path.is_absolute():
        return str(path)
    repo_root = Path(__file__).resolve().parents[1]
    return str(repo_root / path)


def default_run_id() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def resolve_run_dir(run_id: Optional[str], base_dir: str = "runs") -> Tuple[str, str]:
    resolved_run_id = run_id or default_run_id()
    run_dir = resolve_out_dir(os.path.join(base_dir, resolved_run_id))
    return resolved_run_id, run_dir


def prepare_run_dirs(run_id: Optional[str], base_dir: str = "runs") -> Dict[str, str]:
    resolved_run_id, run_dir = resolve_run_dir(run_id, base_dir)
    artifacts_dir = os.path.join(run_dir, "artifacts")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    return {
        "run_id": resolved_run_id,
        "run_dir": run_dir,
        "artifacts_dir": artifacts_dir,
        "logs_dir": logs_dir,
    }


def step5_paths(out_dir: str, model: str, layer: int, n_trials: int) -> Dict[str, str]:
    out_dir = resolve_out_dir(out_dir)
    safe_model = safe_model_name(model)
    suffix = f"{safe_model}_layer{layer}_n{n_trials}"
    return {
        "mean_path": os.path.join(out_dir, f"mean_activations_{suffix}.pt"),
        "fv_path": os.path.join(out_dir, f"fv_{suffix}.pt"),
        "metadata_path": os.path.join(out_dir, f"metadata_{suffix}.json"),
        "suffix": suffix,
    }


def step6_paths(out_dir: str, model: str, layer: int, n_eval: int) -> Dict[str, str]:
    out_dir = resolve_out_dir(out_dir)
    safe_model = safe_model_name(model)
    return {
        "results_path": os.path.join(
            out_dir, f"step6_results_{safe_model}_layer{layer}_n{n_eval}.json"
        ),
        "metadata_path": os.path.join(
            out_dir, f"metadata_step6_{safe_model}_layer{layer}_n{n_eval}.json"
        ),
    }


def save_json(path: str, data: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows provided for CSV save")
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_step5_artifacts(
    out_dir: str,
    model: str,
    layer: int,
    n_trials: int,
    mean_activations,
    fv,
    metadata: Dict[str, object],
) -> Dict[str, str]:
    import torch

    resolved_out_dir = resolve_out_dir(out_dir)
    os.makedirs(resolved_out_dir, exist_ok=True)
    paths = step5_paths(resolved_out_dir, model, layer, n_trials)
    torch.save(mean_activations.cpu(), paths["mean_path"])
    torch.save(fv.cpu(), paths["fv_path"])
    save_json(paths["metadata_path"], metadata)
    return paths


def save_step6_results(
    out_dir: str,
    model: str,
    layer: int,
    n_eval: int,
    payload: Dict[str, object],
    metadata: Dict[str, object],
) -> Dict[str, str]:
    resolved_out_dir = resolve_out_dir(out_dir)
    os.makedirs(resolved_out_dir, exist_ok=True)
    paths = step6_paths(resolved_out_dir, model, layer, n_eval)
    save_json(paths["metadata_path"], metadata)
    save_json(paths["results_path"], payload)
    return paths


def infer_step5_metadata_path(fv_path: str) -> Optional[str]:
    base = os.path.basename(fv_path)
    if base.startswith("fv_") and base.endswith(".pt"):
        suffix = base[len("fv_") : -3]
        return os.path.join(os.path.dirname(fv_path), f"metadata_{suffix}.json")
    return None
