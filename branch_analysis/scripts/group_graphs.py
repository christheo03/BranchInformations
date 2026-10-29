"""
Graphs about the identical-feature groups, as png files in figures/.

    1. group_sizes.png        - how many groups there are of each size.
    2. group_error.png        - how few groups hold most of the groups' error.
    3. group_pure_vs_mixed.png - groups where every branch goes the same way
                                 (pure) against groups that go both ways (mixed).
    4. total_vs_groups.png    - each benchmark's total error, split into the
                                 part inside groups and the part outside them.
    5. total_vs_other_benchmarks.png - each benchmark's total error, and the part caused
                                 by branches that have a twin in a different benchmark.
    6. wmr_size_<range>.png   - graphs, one per group size range: each benchmark's
                                 total WMR and the part caused by groups of that size.

Reads data/identical_br.csv, data/br_to_groupid.csv and data/benchmark_wmr.csv.

Run from branch_analysis/:
    python scripts/group_graphs.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # branch_analysis/
DATA_DIR = os.path.join(ROOT, "data")
FIGURES_DIR = os.path.join(ROOT, "figures")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GRAY = "#898781"


def load_groups():
    groups = pd.read_csv(os.path.join(DATA_DIR, "identical_br.csv"))
    groups["size"] = groups["taken"] + groups["not_taken"]
    groups["mixed"] = (groups["taken"] > 0) & (groups["not_taken"] > 0)
    return groups


def save(fig, name):
    out = os.path.join(FIGURES_DIR, name)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("written", out)


def graph_group_sizes(groups):
    counts = groups["size"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(counts.index, counts.values, color=BLUE, s=25)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Branches in the group (log scale)")
    ax.set_ylabel("Number of groups (log scale)")
    ax.set_title("Most groups are small, a few are huge")
    ax.text(0.97, 0.95,
            f"{len(groups):,} groups, {groups['size'].sum():,} branches\n"
            f"median group size: {int(groups['size'].median())}\n"
            f"{(groups['size'] == 2).mean() * 100:.0f}% of groups are pairs\n"
            f"largest group: {groups['size'].max()} branches",
            transform=ax.transAxes, ha="right", va="top")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "group_sizes.png")


def graph_group_error(groups, total_error):
    ranked = groups.sort_values("wmr_sum", ascending=False)
    cumulative = 100 * ranked["wmr_sum"].cumsum().to_numpy() / ranked["wmr_sum"].sum()
    n = np.arange(1, len(ranked) + 1)
    top100 = cumulative[99]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(n, cumulative, color=BLUE, linewidth=2)
    ax.axvline(100, color=GRAY, linewidth=0.8, linestyle=":")
    ax.annotate(f"top 100 groups: {top100:.0f}% of the error", xy=(100, top100), xytext=(180, top100 - 25),
                arrowprops={"arrowstyle": "->", "color": GRAY})
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Groups, biggest cost first (log scale)")
    ax.set_ylabel("% of the error inside the groups")
    ax.set_title("A hundred groups hold most of the error that sits in groups")
    in_groups = 100 * groups["wmr_sum"].sum() / total_error
    ax.text(0.97, 0.05, f"All groups together hold {in_groups:.0f}% of the model's total error",
            transform=ax.transAxes, ha="right", va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "group_error.png")


def graph_pure_vs_mixed(groups):
    mixed = groups["mixed"]
    measures = {
        "Groups": (groups[~mixed].shape[0], groups[mixed].shape[0]),
        "Branches": (groups.loc[~mixed, "size"].sum(), groups.loc[mixed, "size"].sum()),
        "Error": (groups.loc[~mixed, "wmr_sum"].sum(), groups.loc[mixed, "wmr_sum"].sum()),
    }

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, (pure, mix)) in enumerate(measures.items()):
        total = pure + mix
        ax.barh(i, 100 * pure / total, color=BLUE, label="Pure: every branch goes the same way" if i == 0 else None)
        ax.barh(i, 100 * mix / total, left=100 * pure / total, color=ORANGE, label="Mixed: branches go both ways" if i == 0 else None)
        ax.text(100 * pure / total / 2, i, f"{100 * pure / total:.0f}%", ha="center", va="center", color="white", fontweight="bold")
        ax.text(100 * pure / total + 100 * mix / total / 2, i, f"{100 * mix / total:.0f}%", ha="center", va="center", color="white", fontweight="bold")
    ax.set_yticks(range(len(measures)))
    ax.set_yticklabels(measures.keys())
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the total (%)")
    ax.set_title("Most of the error is in groups where every branch goes the same way")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "group_pure_vs_mixed.png")


def graph_total_vs_groups(groups):
    totals = pd.read_csv(os.path.join(DATA_DIR, "benchmark_wmr.csv")).set_index("benchmark")["wmr"]
    # the groups csv has one column per benchmark: wmr_<benchmark>
    in_groups = pd.Series({bench: groups[f"wmr_{bench}"].sum() for bench in totals.index})
    outside = totals - in_groups
    order = totals.sort_values().index

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(order))
    ax.barh(y, outside[order], color=BLUE, label="Branches with no twin")
    ax.barh(y, in_groups[order], left=outside[order], color=ORANGE, label="Branches inside groups")
    for i, bench in enumerate(order):
        ax.text(totals[bench] + 0.4, i, f"{100 * in_groups[bench] / totals[bench]:.0f}% in groups", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([b.split(".", 1)[1].replace("_r", "") for b in order])
    ax.set_xlabel("Weighted miss rate of the benchmark")
    ax.set_xlim(0, totals.max() * 1.25)
    ax.set_title(f"Where the model's error comes from: {100 * in_groups.sum() / totals.sum():.0f}% of it is inside groups")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "total_vs_groups.png")


def graph_total_vs_other_benchmarks(groups):
    # A group that has branches in 2+ benchmarks: every branch in it has a twin
    # in a different benchmark (even if it also has twins in its own benchmark).
    membership = pd.read_csv(os.path.join(DATA_DIR, "br_to_groupid.csv"))
    spans = membership.groupby("group")["benchmark"].nunique()
    other = groups["group"].map(spans) > 1

    totals = pd.read_csv(os.path.join(DATA_DIR, "benchmark_wmr.csv")).set_index("benchmark")["wmr"]
    cross = pd.Series({bench: groups.loc[other, f"wmr_{bench}"].sum() for bench in totals.index})
    same_only = pd.Series({bench: groups.loc[~other, f"wmr_{bench}"].sum() for bench in totals.index})
    no_twin = totals - cross - same_only
    order = totals.sort_values().index

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(order))
    ax.barh(y, no_twin[order], color=BLUE, label="Branches with no twin")
    ax.barh(y, same_only[order], left=no_twin[order], color=GRAY, label="Twins only in the same benchmark")
    ax.barh(y, cross[order], left=(no_twin + same_only)[order], color=ORANGE, label="Has a twin in a different benchmark")
    for i, bench in enumerate(order):
        ax.text(totals[bench] + 0.4, i, f"{cross[bench]:.2f} ({100 * cross[bench] / totals[bench]:.0f}%)", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([b.split(".", 1)[1].replace("_r", "") for b in order])
    ax.set_xlabel("Weighted miss rate of the benchmark")
    ax.set_xlim(0, totals.max() * 1.25)
    ax.set_title(f"Error from branches with a twin in a different benchmark: {cross.sum():.0f} of {totals.sum():.0f} ({100 * cross.sum() / totals.sum():.0f}%)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "total_vs_other_benchmarks.png")


SIZE_RANGES = [
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-9", 5, 9),
    ("3-9", 3, 9),
    ("10-49", 10, 49),
    ("50+", 50, np.inf),
]


def graph_size_range(groups, totals, label, low, high):
    # Each benchmark's total WMR, and how much of it is caused by the groups
    # whose size is in this range (the wmr_<benchmark> columns, added up).
    in_range = groups[(groups["size"] >= low) & (groups["size"] <= high)]
    part = pd.Series({bench: in_range[f"wmr_{bench}"].sum() for bench in totals.index})
    rest = totals - part
    order = totals.sort_values().index

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(order))
    ax.barh(y, rest[order], color=BLUE, label="Everything else")
    ax.barh(y, part[order], left=rest[order], color=ORANGE, label=f"Caused by groups of {label} branches")
    for i, bench in enumerate(order):
        ax.text(totals[bench] + 0.4, i, f"{part[bench]:.2f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([b.split(".", 1)[1].replace("_r", "") for b in order])
    ax.set_xlabel("Weighted miss rate of the benchmark")
    ax.set_xlim(0, totals.max() * 1.15)
    ax.set_title(f"WMR of each benchmark, and the part from groups of {label} branches")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    name = "wmr_size_" + label.replace("+", "plus") + ".png"
    save(fig, name)


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    groups = load_groups()
    benchmarks = pd.read_csv(os.path.join(DATA_DIR, "benchmark_wmr.csv"))["wmr"]
    total_error = benchmarks.sum()

    graph_group_sizes(groups)
    graph_group_error(groups, total_error)
    graph_pure_vs_mixed(groups)
    graph_total_vs_groups(groups)
    graph_total_vs_other_benchmarks(groups)
    totals = pd.read_csv(os.path.join(DATA_DIR, "benchmark_wmr.csv")).set_index("benchmark")["wmr"]
    for label, low, high in SIZE_RANGES:
        graph_size_range(groups, totals, label, low, high)


if __name__ == "__main__":
    main()
