#!/usr/bin/env python3
"""Plot StepF sweep curves from saved CSV/JSON artifacts.

Dummy data setup:
python - <<'PY'
import csv, json, os
from pathlib import Path

run_id="__dummy_stepF"
art=Path("runs")/run_id/"artifacts"
art.mkdir(parents=True, exist_ok=True)

# summary
(art/"stepF_summary.json").write_text(json.dumps({
  "k_star":10, "best_layer":16, "edit_layer_default":11, "score_key":"mean_delta_p"
}, indent=2))

# k_sweep_L3_results.csv
with open(art/"k_sweep_L3_results.csv","w",newline="") as f:
  w=csv.DictWriter(f, fieldnames=["k","score"])
  w.writeheader()
  for k,s in [(5,0.01),(10,0.03),(20,0.031),(40,0.032)]:
    w.writerow({"k":k,"score":s})

# layer_sweep_results.csv
with open(art/"layer_sweep_results.csv","w",newline="") as f:
  w=csv.DictWriter(f, fieldnames=["edit_layer","score"])
  w.writeheader()
  for l,s in [(0,0.01),(4,0.015),(8,0.02),(12,0.025),(16,0.04),(20,0.035)]:
    w.writerow({"edit_layer":l,"score":s})

# k_sweep_best_layer_results.csv
with open(art/"k_sweep_best_layer_results.csv","w",newline="") as f:
  w=csv.DictWriter(f, fieldnames=["k","score"])
  w.writeheader()
  for k,s in [(1,0.005),(2,0.01),(3,0.02),(5,0.03),(8,0.038),(10,0.04),(20,0.041)]:
    w.writerow({"k":k,"score":s})

print("dummy artifacts at", art)
PY
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fv.io import resolve_out_dir, resolve_run_dir


def load_rows_from_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_rows_from_json(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"JSON file missing 'rows' list: {path}")
    return rows


def find_artifact(base_dir: str, stem: str) -> Tuple[str, str]:
    csv_path = os.path.join(base_dir, f"{stem}.csv")
    json_path = os.path.join(base_dir, f"{stem}.json")
    if os.path.exists(csv_path):
        return csv_path, "csv"
    if os.path.exists(json_path):
        return json_path, "json"
    raise FileNotFoundError(
        f"Missing artifact for {stem}. Expected {csv_path} or {json_path}"
    )


def get_score(row: Dict[str, object], score_key: str) -> float:
    if "score" in row:
        return float(row["score"])
    if score_key in row:
        return float(row[score_key])
    keys = sorted(list(row.keys()))
    raise ValueError(f"Missing score in row. Tried 'score' or '{score_key}'. Keys: {keys}")


def get_layer_value(row: Dict[str, object]) -> int:
    if "layer" in row:
        return int(row["layer"])
    if "edit_layer" in row:
        return int(row["edit_layer"])
    keys = sorted(list(row.keys()))
    raise ValueError(f"Missing layer in row. Tried 'layer' or 'edit_layer'. Keys: {keys}")


def plot_curve(
    x_vals: List[int],
    y_vals: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    marker: Optional[str],
    dpi: int,
    vline_x: Optional[float] = None,
    vline_label: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=dpi)
    ax.plot(x_vals, y_vals, marker=marker if marker else None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if vline_x is not None:
        ax.axvline(vline_x, color="red", linestyle="--", linewidth=1.0)
        if vline_label:
            ax.text(vline_x, max(y_vals), vline_label, color="red", ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot StepF sweep curves.")
    parser.add_argument("--run_id_stepF", required=True, help="StepF run_id (required)")
    parser.add_argument(
        "--score_key",
        default="mean_delta_logprob",
        choices=["delta_acc", "mean_delta_logprob", "mean_delta_p", "mean_delta_logit"],
        help="Score key (default: mean_delta_logprob)",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf"],
        help="Output format (default: png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output dpi (default: 150)")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/<run_id_stepF>/artifacts/plots)",
    )
    parser.add_argument(
        "--no_markers",
        action="store_true",
        help="Disable point markers on plots",
    )
    args = parser.parse_args()

    resolved_run_id, run_dir = resolve_run_dir(args.run_id_stepF)
    artifacts_dir = os.path.join(run_dir, "artifacts")
    if args.out_dir:
        out_dir = resolve_out_dir(args.out_dir)
    else:
        out_dir = os.path.join(artifacts_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    try:
        k_sweep_path, k_sweep_type = find_artifact(artifacts_dir, "k_sweep_L3_results")
    except FileNotFoundError as exc:
        print(str(exc))
        return 1
    try:
        layer_sweep_path, layer_sweep_type = find_artifact(artifacts_dir, "layer_sweep_results")
    except FileNotFoundError as exc:
        print(str(exc))
        return 1

    best_k_path = None
    best_k_type = None
    try:
        best_k_path, best_k_type = find_artifact(
            artifacts_dir, "k_sweep_best_layer_results"
        )
    except FileNotFoundError:
        best_k_path = None

    summary_path = os.path.join(artifacts_dir, "stepF_summary.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    def load_rows(path: str, kind: str) -> List[Dict[str, object]]:
        if kind == "csv":
            return load_rows_from_csv(path)
        return load_rows_from_json(path)

    k_rows = load_rows(k_sweep_path, k_sweep_type)
    layer_rows = load_rows(layer_sweep_path, layer_sweep_type)
    best_k_rows = load_rows(best_k_path, best_k_type) if best_k_path else None

    k_points = sorted(
        [(int(row["k"]), get_score(row, args.score_key)) for row in k_rows],
        key=lambda x: x[0],
    )
    layer_points = sorted(
        [(get_layer_value(row), get_score(row, args.score_key)) for row in layer_rows],
        key=lambda x: x[0],
    )

    marker = None if args.no_markers else "o"
    k_star = summary.get("k_star") if isinstance(summary, dict) else None
    best_layer = summary.get("best_layer") if isinstance(summary, dict) else None

    k_curve_path = os.path.join(out_dir, f"k_curve_L3.{args.format}")
    plot_curve(
        [x for x, _ in k_points],
        [y for _, y in k_points],
        title=f"k sweep @ L/3 (score={args.score_key})",
        xlabel="k",
        ylabel=args.score_key,
        out_path=k_curve_path,
        marker=marker,
        dpi=args.dpi,
        vline_x=k_star,
        vline_label="k*" if k_star is not None else None,
    )
    print(f"saved: {k_curve_path}")

    layer_curve_path = os.path.join(out_dir, f"layer_curve_kstar.{args.format}")
    plot_curve(
        [x for x, _ in layer_points],
        [y for _, y in layer_points],
        title=f"layer sweep @ k* (score={args.score_key})",
        xlabel="edit_layer",
        ylabel=args.score_key,
        out_path=layer_curve_path,
        marker=marker,
        dpi=args.dpi,
        vline_x=best_layer,
        vline_label="best_layer" if best_layer is not None else None,
    )
    print(f"saved: {layer_curve_path}")

    if best_k_rows:
        best_k_points = sorted(
            [(int(row["k"]), get_score(row, args.score_key)) for row in best_k_rows],
            key=lambda x: x[0],
        )
        best_k_curve_path = os.path.join(out_dir, f"k_curve_best_layer.{args.format}")
        plot_curve(
            [x for x, _ in best_k_points],
            [y for _, y in best_k_points],
            title=f"k sweep @ best_layer (score={args.score_key})",
            xlabel="k",
            ylabel=args.score_key,
            out_path=best_k_curve_path,
            marker=marker,
            dpi=args.dpi,
        )
        print(f"saved: {best_k_curve_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
