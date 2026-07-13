"""Render a grouped bar chart for one metric from a results CSV.

Unlike the original notebook (which had result numbers typed into the plot
code by hand), this reads the CSV produced by holdout_test_eval.py /
kfold_cv_eval.py, so the chart always matches the current results file.

Usage:
    python -m evaluation.plot_results \
        --csv evaluation/results/holdout_results.csv \
        --metric mAP50 --out docs/assets/map50_comparison.png
        [--font "TH Sarabun New"]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(df: pd.DataFrame, metric: str, out_path: Path, font: str = None) -> None:
    if font:
        plt.rcParams["font.family"] = font

    labels = df["Model"].tolist()
    values = df[metric].tolist()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color="#4C72B0", edgecolor="black", width=0.6)

    plt.xlabel("Model", fontsize=16, labelpad=10, fontweight="bold")
    plt.ylabel(metric, fontsize=16, labelpad=10, fontweight="bold")
    plt.xticks(fontsize=12, rotation=30, ha="right")
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f"{yval:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", type=Path, required=True,
                        help="results CSV with a 'Model' column and metric columns")
    parser.add_argument("--metric", default="mAP50")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--font", help='e.g. "TH Sarabun New" to reproduce Thai-labeled figures')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    # drop aggregate rows (MEAN/SD/AVERAGE) so they don't appear as models
    if "Fold" in df.columns:
        df = df[~df["Fold"].astype(str).isin(["MEAN", "SD", "AVERAGE"])]

    if args.metric not in df.columns:
        raise SystemExit(f"ERROR: column '{args.metric}' not in {args.csv} "
                         f"(available: {list(df.columns)})")

    plot_metric(df, args.metric, args.out, args.font)


if __name__ == "__main__":
    main()
