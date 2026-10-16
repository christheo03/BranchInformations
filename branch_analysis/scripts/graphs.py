"""
Make the graphs for the presentation, as png files in figures/.

    1. wmr_vs_perfect.png   - each benchmark's weighted miss rate (WMR) against
                              the perfect predictor's.
    2. error_concentration.png - how few mispredicted branches account for most
                              of a benchmark's WMR.

Run from branch_analysis/:
    python scripts/graphs.py
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # branch_analysis/
sys.path.insert(0, ROOT)
from helper import branch_wmr  # noqa: E402

PREDICTIONS_DIR = os.path.join(ROOT, "predictions")
DATA_DIR = os.path.join(ROOT, "data")
FIGURES_DIR = os.path.join(ROOT, "figures")

MODEL_COLOR = "#2a78d6"
PERFECT_COLOR = "#eb6834"
GRAY = "#898781"


def short(name):
    # 502.gcc_r -> gcc
    return name.split(".", 1)[1].replace("_r", "")


def graph_wmr_vs_perfect():
    d = pd.read_csv(os.path.join(DATA_DIR, "benchmark_wmr.csv")).sort_values("wmr", ascending=False)
    x = np.arange(len(d))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - width / 2, d["wmr"], width, label=f"Model (mean {d['wmr'].mean():.1f})", color=MODEL_COLOR)
    ax.bar(x + width / 2, d["perfect_wmr"], width, label=f"Perfect predictor (mean {d['perfect_wmr'].mean():.1f})", color=PERFECT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([short(b) for b in d["benchmark"]], rotation=45, ha="right")
    ax.set_ylabel("Weighted miss rate")
    ax.set_title("Weighted miss rate per benchmark: model vs perfect predictor")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "wmr_vs_perfect.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("written", out)


def graph_error_concentration():
    fig, ax = plt.subplots(figsize=(10, 6))
    half = []  # percentage of mispredicted branches needed for 50% of the error

    for path in sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "EMB_CT_*.csv"))):
        bench = os.path.basename(path).replace("EMB_CT_", "").replace(".csv", "")
        df = pd.read_csv(path, usecols=["rate", "weight", "predicted", "actual"])

        wrong = df[df["predicted"] != df["actual"]]
        errors = branch_wmr(wrong["predicted"], wrong["rate"], wrong["weight"]).sort_values(ascending=False)
        cumulative = 100 * errors.cumsum().to_numpy() / errors.sum()
        branches_pct = 100 * np.arange(1, len(errors) + 1) / len(errors)

        ax.plot(branches_pct, cumulative, color=MODEL_COLOR, alpha=0.35, linewidth=1)
        half.append(branches_pct[np.searchsorted(cumulative, 50)])

    # y = x, drawn with many points: on a log axis a two-point line would be straight on screen and wrong.
    equal = np.logspace(-2, 2, 200)
    ax.plot(equal, equal, color=GRAY, linestyle="--", label="Every branch costs the same")
    ax.axhline(50, color=GRAY, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of mispredicted branches (biggest cost first, log scale)")
    ax.set_ylabel("% of the benchmark's misclassification WMR")
    ax.set_title("A few mispredicted branches carry most of the error")
    ax.plot([], [], color=MODEL_COLOR, alpha=0.6, label=f"One line per benchmark (median: {np.median(half):.1f}% of branches reach 50%)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "error_concentration.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("written", out)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    graph_wmr_vs_perfect()
    graph_error_concentration()


if __name__ == "__main__":
    main()
