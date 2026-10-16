"""
Look up one branch by benchmark and PC.

Run from branch_analysis/:
    python lookup.py 502.gcc_r 0x6718be
    python lookup.py gcc 0x6718be        (the short name is enough)
"""
import argparse
import os

import pandas as pd

from helper import branch_wmr_if_wrong

ROOT = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(ROOT, "predictions")
DATA_DIR = os.path.join(ROOT, "data")

WIDTH = 64
LABEL_WIDTH = 26


def find_benchmarks(text):
    # "imagick" or "538.imagick_r" -> the full benchmark names that match
    names = sorted(
        f.replace("EMB_CT_", "").replace(".csv", "")
        for f in os.listdir(PREDICTIONS_DIR) if f.startswith("EMB_CT_")
    )
    if text in names:
        return [text]
    return [n for n in names if text.lower() in n.lower()]


def direction(value):
    return "Taken" if value == 1 else "NotTaken"


def line(label, value):
    print(f"  {label:<{LABEL_WIDTH}}{value}")


def section(title):
    print(f"\n {title}")
    print(" " + "-" * (WIDTH - 2))


def print_branch(branch):
    section("BRANCH")
    line("Opcode", branch["Opcode"])
    line("Routine type", branch["Routine_Type"])
    line("Actual", direction(branch["actual"]))
    line("Prediction", direction(branch["predicted"]))
    line("Executed", f"{int(branch['Executed']):,}")
    line("Dynamic taken", f"{int(branch['Taken']):,}")
    line("Dynamic not taken", f"{int(branch['Executed'] - branch['Taken']):,}")
    line("Taken rate", f"{100 * branch['rate']:.2f}%")
    biased = branch["rate"] <= 0.01 or branch["rate"] >= 0.99
    line("Highly biased (99%)", "yes" if biased else "no")


def print_cost(branch, benchmark_wmr):
    # The weighted miss rate of the branch: its weighted error when it is
    # predicted wrong, whether or not the model got it right. 0-100 scale.
    wmr = branch_wmr_if_wrong(branch["actual"], branch["rate"], branch["weight"])

    section("COST")
    line("Weight in benchmark", f"{branch['weight']:.3e}  ({100 * branch['weight']:.6f}% of the benchmark)")
    line("Weighted miss rate", f"{100 * wmr:.6f}")

    # Only when the model got it wrong: how much of the benchmark's weighted
    # miss rate this branch is responsible for.
    if branch["predicted"] != branch["actual"]:
        line("Share of benchmark WMR", f"{100 * wmr / benchmark_wmr:.4f}%  (benchmark WMR {benchmark_wmr:.2f})")


def print_group(bench, pc):
    section("IDENTICAL-FEATURE GROUP")

    membership = pd.read_csv(os.path.join(DATA_DIR, "br_to_groupid.csv"))
    match = membership[(membership["benchmark"] == bench) & (membership["pc"].str.lower() == pc.lower())]
    if match.empty:
        line("Group", "none (no other branch has exactly the same features)")
        return

    gid = match.iloc[0]["group"]
    groups = pd.read_csv(os.path.join(DATA_DIR, "identical_br.csv"))
    g = groups[groups["group"] == gid].iloc[0]

    taken, not_taken = int(g["taken"]), int(g["not_taken"])
    rank = int((groups["wmr_sum"] > g["wmr_sum"]).sum()) + 1
    line("Group", f"{gid}")
    line("Branches in the group", f"{taken + not_taken:,}")
    line("Go taken / not taken", f"{taken:,} / {not_taken:,}")
    line("Predicted right / wrong", f"{int(g['predicted_correct']):,} / {int(g['predicted_wrong']):,}")
    line("Group weighted miss rate", f"{g['wmr_sum']:.4f}  (rank {rank:,} of {len(groups):,} groups)")


def print_benchmark(benchmark):
    section("BENCHMARK")
    line("Weighted miss rate", f"{benchmark['wmr']:.2f}")
    line("Perfect predictor", f"{benchmark['perfect_wmr']:.2f}")
    line("Branches", f"{int(benchmark['branches']):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("pc")
    args = parser.parse_args()
    pc = args.pc

    matches = find_benchmarks(args.benchmark)
    if not matches:
        print(f"Benchmark '{args.benchmark}' does not exist.")
        return
    if len(matches) > 1:
        print(f"'{args.benchmark}' matches more than one benchmark: {', '.join(matches)}")
        return
    bench = matches[0]

    path = os.path.join(PREDICTIONS_DIR, f"EMB_CT_{bench}.csv")

    columns = ["Address", "Opcode", "Routine_Type", "Executed", "Taken", "rate", "weight", "predicted", "actual"]
    df = pd.read_csv(path, usecols=columns)
    rows = df[df["Address"].str.lower() == pc.lower()]
    if rows.empty:
        print(f"PC {pc} does not exist in {bench}.")
        return

    branch = rows.iloc[0]
    benchmark = pd.read_csv(os.path.join(DATA_DIR, "benchmark_wmr.csv")).set_index("benchmark").loc[bench]

    print("=" * WIDTH)
    print(f" Benchmark: {bench}      PC: {pc}")
    print("=" * WIDTH)
    print_branch(branch)
    print_cost(branch, benchmark["wmr"])
    print_group(bench, pc)
    print_benchmark(benchmark)
    print("\n" + "=" * WIDTH)


if __name__ == "__main__":
    main()
