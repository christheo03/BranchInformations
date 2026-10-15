import argparse
import os

import pandas as pd

from helper import branch_wmr_if_wrong

ROOT = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(ROOT, "predictions")
BENCHMARK_WMR_FILE = os.path.join(ROOT, "data", "benchmark_wmr.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark")
    parser.add_argument("pc")
    args = parser.parse_args()
    bench = args.benchmark
    pc = args.pc

    

    path = os.path.join(PREDICTIONS_DIR, f"EMB_CT_{bench}.csv")
    if not os.path.isfile(path):
        print(f"Benchmark '{bench}' does not exist.")
        return

    df = pd.read_csv(path, usecols=["Address", "Executed", "Taken", "rate", "weight", "predicted", "actual"])
    rows = df[df["Address"].str.lower() == pc.lower()]
    if rows.empty:
        print(f"PC {pc} does not exist in {bench}.")
        return

    print(f"Benchmark: {bench} PC: {pc}")
    branch = rows.iloc[0]
    actual = "Taken" if branch["actual"] == 1 else "NotTaken"
    predicted = "Taken" if branch["predicted"] == 1 else "NotTaken"
    print(f"Actual: {actual}")
    print(f"Prediction: {predicted}")
    print(f"Dynamic taken: {branch['Taken']}")
    print(f"Dynamic not taken: {branch['Executed'] - branch['Taken']}")
    print(f"Weight in benchmark: {branch['weight']:.3e} ({100 * branch['weight']:.6f}% of the benchmark)")

    # The weighted miss rate of the branch: its weighted error when it is
    # predicted wrong, whether or not the model got it right. 0-100 scale.
    wmr = branch_wmr_if_wrong(branch["actual"], branch["rate"], branch["weight"])
    print(f"Weighted miss rate: {100 * wmr:.6f}")

    # Only when the model got it wrong: how much of the benchmark's weighted
    # miss rate this branch is responsible for.
    if branch["predicted"] != branch["actual"]:
        benchmark_wmr = pd.read_csv(BENCHMARK_WMR_FILE).set_index("benchmark").loc[bench, "wmr"]
        print(f"Share of the benchmark's weighted miss rate ({benchmark_wmr:.2f}): {100 * wmr / benchmark_wmr:.4f}%")


if __name__ == "__main__":
    main()
