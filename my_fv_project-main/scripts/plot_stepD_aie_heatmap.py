#!/usr/bin/env python3
"""Plot StepD AIE heatmap.

Example:
  python scripts/plot_stepD_aie_heatmap.py --run_id stepD_llama31_8b_smoke --metric mean_cie --topk 20
"""

import argparse
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot StepD AIE heatmap from aie_scores.csv.")
    parser.add_argument("--run_id", required=True, help="StepD run id (required)")
    parser.add_argument("--metric", default="mean_cie", help="Metric column (default: mean_cie)")
    parser.add_argument(
        "--out_name",
        default="aie_heatmap_{metric}.png",
        help="Output filename (default: aie_heatmap_{metric}.png)",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    parser.add_argument("--vmin", type=float, default=None, help="Color scale min")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale max")
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Overlay top-k points by metric (default: 0 = off)",
    )
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--dpi", type=int, default=200, help="Output dpi (default: 200)")
    return parser.parse_args()


def _suggest_numeric_columns(df: pd.DataFrame) -> List[str]:
    candidates = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    return candidates[:5]


def _safe_int_series(series: pd.Series) -> pd.Series:
    try:
        return series.astype(int)
    except Exception:
        return series


def _tick_positions(values: List[int], max_ticks: int = 20) -> Tuple[List[int], List[str]]:
    if not values:
        return [], []
    if len(values) <= max_ticks:
        return list(range(len(values))), [str(v) for v in values]
    step = max(1, len(values) // max_ticks)
    idxs = list(range(0, len(values), step))
    return idxs, [str(values[i]) for i in idxs]


def main() -> int:
    args = parse_args()

    csv_path = os.path.join("runs", args.run_id, "artifacts", "aie_scores.csv")
    if not os.path.exists(csv_path):
        print(f"Missing aie_scores.csv at: {csv_path}")
        return 1

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"Failed to read CSV at {csv_path}: {exc}")
        return 1

    required_cols = {"layer", "head", args.metric}
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        suggestions = _suggest_numeric_columns(df)
        suggestion_msg = ""
        if suggestions:
            suggestion_msg = f" Available numeric columns: {', '.join(suggestions)}"
        print(f"Missing columns in CSV: {', '.join(missing)}.{suggestion_msg}")
        return 1

    df = df.copy()
    df["layer"] = _safe_int_series(df["layer"])
    df["head"] = _safe_int_series(df["head"])
    df = df.sort_values(["layer", "head"])

    try:
        pivot = df.pivot(index="layer", columns="head", values=args.metric)
    except Exception as exc:
        print(f"Failed to pivot CSV: {exc}")
        return 1

    layers = list(pivot.index)
    heads = list(pivot.columns)
    matrix = pivot.values

    if args.out_name == "aie_heatmap_{metric}.png":
        out_name = args.out_name.format(metric=args.metric)
    else:
        out_name = args.out_name
    out_path = os.path.join("runs", args.run_id, "artifacts", out_name)

    title = args.title
    if title is None:
        title = f"AIE heatmap ({args.metric})"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=args.dpi)
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(args.metric)

    x_ticks, x_labels = _tick_positions(heads)
    y_ticks, y_labels = _tick_positions(layers)
    ax.set_xticks(x_ticks, x_labels)
    ax.set_yticks(y_ticks, y_labels)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(title)

    if args.topk and args.topk > 0:
        topk = df.sort_values(args.metric, ascending=False).head(args.topk)
        head_to_idx = {head: i for i, head in enumerate(heads)}
        layer_to_idx = {layer: i for i, layer in enumerate(layers)}
        xs = [head_to_idx[row["head"]] for _, row in topk.iterrows() if row["head"] in head_to_idx]
        ys = [layer_to_idx[row["layer"]] for _, row in topk.iterrows() if row["layer"] in layer_to_idx]
        ax.scatter(xs, ys, s=20, c="white", marker="o", edgecolors="black", linewidths=0.5)

    fig.tight_layout()
    try:
        fig.savefig(out_path)
    except Exception as exc:
        print(f"Failed to save figure to {out_path}: {exc}")
        return 1
    finally:
        plt.close(fig)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
