#!/usr/bin/env python3
"""STEP F: Run k-sweep, layer-sweep, then k-sweep again via StepE/Step6.

Example:
  python scripts/run_stepF_k_then_layer_sweep.py --run_id_stepD stepD_smoke --model gpt2 --model_spec gpt2
"""

import argparse
import os
import shlex
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.io import prepare_run_dirs, resolve_out_dir, save_csv, save_json, load_json


def make_logger(log_path: str):
    log_file = open(log_path, "w", encoding="utf-8")

    def log(message: str) -> None:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()

    return log, log_file


def parse_int_list(value: str) -> List[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def build_command(base: List[str], extra: Dict[str, Optional[str]]) -> List[str]:
    cmd = list(base)
    for key, value in extra.items():
        if value is None:
            continue
        if value is True:
            cmd.append(key)
            continue
        if value is False:
            continue
        cmd.append(key)
        cmd.append(str(value))
    return cmd


def format_command(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def find_step6_results(step6_run_id: str) -> str:
    results_dir = resolve_out_dir(os.path.join("runs", step6_run_id, "artifacts", "step6"))
    matches = glob(os.path.join(results_dir, "step6_results_*.json"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 step6_results_*.json in {results_dir}, found {len(matches)}"
        )
    return matches[0]


def load_step6_score(results_path: str, score_key: str) -> float:
    payload = load_json(results_path)
    summary = payload.get("summary", {})
    if score_key not in summary:
        raise ValueError(f"Missing score_key '{score_key}' in summary of {results_path}")
    return float(summary[score_key])


def run_subprocess(cmd: List[str], log) -> None:
    log(f"exec: {format_command(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="STEP F: k-sweep then layer-sweep orchestration.")
    parser.add_argument("--run_id_stepD", required=True, help="StepD run_id (required)")
    parser.add_argument("--model", default="gpt2", help="Model name or path (default: gpt2)")
    parser.add_argument(
        "--model_spec",
        default="gpt2",
        help="Model spec name for adapter resolution (default: gpt2)",
    )
    parser.add_argument(
        "--dataset_path",
        default="datasets/processed/antonym.json",
        help="Path to antonym dataset JSON",
    )
    parser.add_argument(
        "--metric",
        default="mean_cie",
        choices=["mean_cie", "mean_abs_cie"],
        help="AIE ranking score column (default: mean_cie)",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Injection scale (default: 1.0)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--n_eval", type=int, default=20, help="Step6 eval prompts")
    parser.add_argument(
        "--score_key",
        default="mean_delta_logprob",
        choices=["delta_acc", "mean_delta_logprob", "mean_delta_p", "mean_delta_logit"],
        help="Score key from Step6 summary (default: mean_delta_logprob)",
    )
    parser.add_argument(
        "--k_list_coarse",
        default="5,10,20,40",
        help="Comma-separated k list for coarse sweep",
    )
    parser.add_argument(
        "--layers_to_try",
        default="0,4,8,12,16,20,24,28,31",
        help="Comma-separated layer list for coarse sweep",
    )
    parser.add_argument("--k_star", type=int, default=None, help="Override plateau k*")
    parser.add_argument(
        "--plateau_eps",
        type=float,
        default=0.001,
        help="k* selection epsilon (default: 0.001)",
    )
    parser.add_argument(
        "--k_list_fine",
        default=None,
        help="Optional comma-separated k list for fine sweep",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "cuda", "auto"],
        help="Device selection (default: auto)",
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
    parser.add_argument("--run_id", default=None, help="StepF run_id (default: auto)")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands and output paths without executing",
    )
    args = parser.parse_args()

    run_info = prepare_run_dirs(args.run_id)
    log_path = os.path.join(run_info["logs_dir"], "stepF.log")
    log, log_file = make_logger(log_path)

    log("stepF start")
    log(f"run_id: {run_info['run_id']}")
    log(f"log_path: {log_path}")
    log(f"score_key: {args.score_key}")

    stepd_dir = resolve_out_dir(os.path.join("runs", args.run_id_stepD, "artifacts"))
    clean_mean_path = os.path.join(stepd_dir, "clean_mean.pt")
    if not os.path.exists(clean_mean_path):
        log(f"Missing clean_mean.pt: {clean_mean_path}")
        log_file.close()
        return 1

    try:
        import torch
    except Exception as exc:  # pragma: no cover - runtime import check
        log(f"Failed to import torch: {exc}")
        log_file.close()
        return 1

    blob = torch.load(clean_mean_path, map_location="cpu")
    layers_blob = blob.get("layers")
    if not layers_blob:
        log("clean_mean.pt missing 'layers' list")
        log_file.close()
        return 1
    max_layer = max(int(layer) for layer in layers_blob)
    L = max_layer + 1
    edit_layer_default = round(L / 3)
    log(f"L={L} edit_layer_default={edit_layer_default}")

    k_list_coarse = parse_int_list(args.k_list_coarse)
    layers_to_try = parse_int_list(args.layers_to_try)
    k_list_fine = (
        parse_int_list(args.k_list_fine)
        if args.k_list_fine is not None
        else list(k_list_coarse)
    )

    artifact_paths = [
        os.path.join(run_info["artifacts_dir"], "k_sweep_L3_results.json"),
        os.path.join(run_info["artifacts_dir"], "k_sweep_L3_results.csv"),
        os.path.join(run_info["artifacts_dir"], "layer_sweep_results.json"),
        os.path.join(run_info["artifacts_dir"], "layer_sweep_results.csv"),
        os.path.join(run_info["artifacts_dir"], "k_sweep_best_layer_results.json"),
        os.path.join(run_info["artifacts_dir"], "k_sweep_best_layer_results.csv"),
        os.path.join(run_info["artifacts_dir"], "stepF_summary.json"),
    ]

    if args.dry_run:
        log("dry_run enabled; commands will not be executed")
        log("planned outputs:")
        for path in artifact_paths:
            log(f"- {path}")

    common_stepE_args = {
        "--model": args.model,
        "--model_spec": args.model_spec,
        "--dataset_path": args.dataset_path,
        "--metric": args.metric,
        "--alpha": args.alpha,
        "--seed": args.seed,
        "--quant": args.quant,
        "--device": args.device,
        "--dtype": args.dtype,
        "--device_map": args.device_map,
        "--skip_eval": True,
    }
    common_step6_args = {
        "--model": args.model,
        "--model_spec": args.model_spec,
        "--dataset_path": args.dataset_path,
        "--alpha": args.alpha,
        "--seed": args.seed,
        "--n_eval": args.n_eval,
        "--quant": args.quant,
        "--device": args.device,
        "--dtype": args.dtype,
        "--device_map": args.device_map,
        "--score_key": args.score_key,
    }

    k_sweep_rows = []
    for k in k_list_coarse:
        stepE_run_id = f"{run_info['run_id']}__stepE_k{k}"
        step6_run_id = f"{run_info['run_id']}__step6_ksweepL3_k{k}_L{edit_layer_default}"

        cmd_stepE = build_command(
            [
                sys.executable,
                "scripts/run_stepE_topk_fv_and_eval.py",
                "--run_id_stepD",
                args.run_id_stepD,
                "--k",
                str(k),
                "--run_id",
                stepE_run_id,
            ],
            common_stepE_args,
        )
        cmd_step6 = build_command(
            [
                sys.executable,
                "scripts/run_step6_fv_injection_eval.py",
                "--run_id_stepE",
                stepE_run_id,
                "--edit_layer",
                str(edit_layer_default),
                "--run_id",
                step6_run_id,
            ],
            common_step6_args,
        )

        if args.dry_run:
            log(f"[k_sweep_L3] k={k}")
            log(f"stepE: {format_command(cmd_stepE)}")
            log(f"step6: {format_command(cmd_step6)}")
            continue

        try:
            run_subprocess(cmd_stepE, log)
            run_subprocess(cmd_step6, log)
        except subprocess.CalledProcessError as exc:
            log(f"Command failed: {exc}")
            log_file.close()
            raise

        results_path = find_step6_results(step6_run_id)
        score = load_step6_score(results_path, args.score_key)
        k_sweep_rows.append(
            {
                "phase": "k_sweep_L3",
                "k": k,
                "edit_layer": edit_layer_default,
                "score": score,
                "stepE_run_id": stepE_run_id,
                "step6_run_id": step6_run_id,
                "step6_results_path": results_path,
            }
        )

    if not args.dry_run:
        k_sweep_meta = {
            "phase": "k_sweep_L3",
            "run_id_stepD": args.run_id_stepD,
            "model": args.model,
            "model_spec": args.model_spec,
            "dataset_path": args.dataset_path,
            "metric": args.metric,
            "score_key": args.score_key,
            "alpha": args.alpha,
            "seed": args.seed,
            "n_eval": args.n_eval,
            "edit_layer": edit_layer_default,
            "k_list": k_list_coarse,
        }
        k_sweep_payload = {"meta": k_sweep_meta, "rows": k_sweep_rows}
        save_json(os.path.join(run_info["artifacts_dir"], "k_sweep_L3_results.json"), k_sweep_payload)
        save_csv(os.path.join(run_info["artifacts_dir"], "k_sweep_L3_results.csv"), k_sweep_rows)

    if args.dry_run:
        if args.k_star is None:
            k_star = k_list_coarse[0] if k_list_coarse else 0
            log(f"dry_run: using placeholder k_star={k_star}")
        else:
            k_star = args.k_star
        placeholder_layer = layers_to_try[0] if layers_to_try else edit_layer_default
        log(f"dry_run: using placeholder best_layer={placeholder_layer}")
    else:
        k_star = None

    if not args.dry_run:
        if args.k_star is not None:
            k_star = args.k_star
        else:
            if not k_sweep_rows:
                log("No k_sweep rows available to select k_star")
                log_file.close()
                return 1
            best_score = max(row["score"] for row in k_sweep_rows)
            eligible = [
                row["k"]
                for row in k_sweep_rows
                if row["score"] >= best_score - args.plateau_eps
            ]
            if eligible:
                k_star = min(eligible)
            else:
                k_star = max(k_sweep_rows, key=lambda r: r["score"])["k"]

    stepE_run_id = f"{run_info['run_id']}__stepE_kstar{k_star}"
    cmd_stepE = build_command(
        [
            sys.executable,
            "scripts/run_stepE_topk_fv_and_eval.py",
            "--run_id_stepD",
            args.run_id_stepD,
            "--k",
            str(k_star),
            "--run_id",
            stepE_run_id,
        ],
        common_stepE_args,
    )
    if args.dry_run:
        log(f"[layer_sweep] k_star={k_star}")
        log(f"stepE: {format_command(cmd_stepE)}")
    else:
        try:
            run_subprocess(cmd_stepE, log)
        except subprocess.CalledProcessError as exc:
            log(f"Command failed: {exc}")
            log_file.close()
            raise

    layer_rows = []
    for layer in layers_to_try:
        step6_run_id = f"{run_info['run_id']}__step6_layersweep_k{k_star}_L{layer}"
        cmd_step6 = build_command(
            [
                sys.executable,
                "scripts/run_step6_fv_injection_eval.py",
                "--run_id_stepE",
                stepE_run_id,
                "--edit_layer",
                str(layer),
                "--run_id",
                step6_run_id,
            ],
            common_step6_args,
        )
        if args.dry_run:
            log(f"step6: {format_command(cmd_step6)}")
            continue
        try:
            run_subprocess(cmd_step6, log)
        except subprocess.CalledProcessError as exc:
            log(f"Command failed: {exc}")
            log_file.close()
            raise

        results_path = find_step6_results(step6_run_id)
        score = load_step6_score(results_path, args.score_key)
        layer_rows.append(
            {
                "phase": "layer_sweep",
                "k": k_star,
                "edit_layer": layer,
                "score": score,
                "stepE_run_id": stepE_run_id,
                "step6_run_id": step6_run_id,
                "step6_results_path": results_path,
            }
        )

    if args.dry_run:
        best_layer = placeholder_layer
    else:
        layer_meta = {
            "phase": "layer_sweep",
            "run_id_stepD": args.run_id_stepD,
            "model": args.model,
            "model_spec": args.model_spec,
            "dataset_path": args.dataset_path,
            "metric": args.metric,
            "score_key": args.score_key,
            "alpha": args.alpha,
            "seed": args.seed,
            "n_eval": args.n_eval,
            "k_star": k_star,
            "layers_to_try": layers_to_try,
        }
        layer_payload = {"meta": layer_meta, "rows": layer_rows}
        save_json(os.path.join(run_info["artifacts_dir"], "layer_sweep_results.json"), layer_payload)
        save_csv(os.path.join(run_info["artifacts_dir"], "layer_sweep_results.csv"), layer_rows)

        if not layer_rows:
            log("No layer sweep rows available to select best_layer")
            log_file.close()
            return 1
        best_layer = min(
            [row for row in layer_rows if row["score"] == max(r["score"] for r in layer_rows)],
            key=lambda r: r["edit_layer"],
        )["edit_layer"]

    best_layer_rows = []
    for k in k_list_fine:
        stepE_run_id = f"{run_info['run_id']}__stepE_bestL{best_layer}_k{k}"
        step6_run_id = f"{run_info['run_id']}__step6_ksweepBestL_L{best_layer}_k{k}"
        cmd_stepE = build_command(
            [
                sys.executable,
                "scripts/run_stepE_topk_fv_and_eval.py",
                "--run_id_stepD",
                args.run_id_stepD,
                "--k",
                str(k),
                "--run_id",
                stepE_run_id,
            ],
            common_stepE_args,
        )
        cmd_step6 = build_command(
            [
                sys.executable,
                "scripts/run_step6_fv_injection_eval.py",
                "--run_id_stepE",
                stepE_run_id,
                "--edit_layer",
                str(best_layer),
                "--run_id",
                step6_run_id,
            ],
            common_step6_args,
        )
        if args.dry_run:
            log(f"[k_sweep_best_layer] k={k} best_layer={best_layer}")
            log(f"stepE: {format_command(cmd_stepE)}")
            log(f"step6: {format_command(cmd_step6)}")
            continue
        try:
            run_subprocess(cmd_stepE, log)
            run_subprocess(cmd_step6, log)
        except subprocess.CalledProcessError as exc:
            log(f"Command failed: {exc}")
            log_file.close()
            raise

        results_path = find_step6_results(step6_run_id)
        score = load_step6_score(results_path, args.score_key)
        best_layer_rows.append(
            {
                "phase": "k_sweep_best_layer",
                "k": k,
                "edit_layer": best_layer,
                "score": score,
                "stepE_run_id": stepE_run_id,
                "step6_run_id": step6_run_id,
                "step6_results_path": results_path,
            }
        )

    if args.dry_run:
        log_file.close()
        return 0

    best_layer_meta = {
        "phase": "k_sweep_best_layer",
        "run_id_stepD": args.run_id_stepD,
        "model": args.model,
        "model_spec": args.model_spec,
        "dataset_path": args.dataset_path,
        "metric": args.metric,
        "score_key": args.score_key,
        "alpha": args.alpha,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "best_layer": best_layer,
        "k_list": k_list_fine,
    }
    best_layer_payload = {"meta": best_layer_meta, "rows": best_layer_rows}
    save_json(
        os.path.join(run_info["artifacts_dir"], "k_sweep_best_layer_results.json"),
        best_layer_payload,
    )
    save_csv(
        os.path.join(run_info["artifacts_dir"], "k_sweep_best_layer_results.csv"),
        best_layer_rows,
    )

    summary = {
        "stepF_run_id": run_info["run_id"],
        "run_id_stepD": args.run_id_stepD,
        "model": args.model,
        "model_spec": args.model_spec,
        "dataset_path": args.dataset_path,
        "metric": args.metric,
        "score_key": args.score_key,
        "alpha": args.alpha,
        "seed": args.seed,
        "n_eval": args.n_eval,
        "L": L,
        "edit_layer_default": edit_layer_default,
        "k_list_coarse": k_list_coarse,
        "k_star": k_star,
        "layers_to_try": layers_to_try,
        "best_layer": best_layer,
        "k_list_fine": k_list_fine,
    }
    save_json(os.path.join(run_info["artifacts_dir"], "stepF_summary.json"), summary)

    log("stepF complete")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
