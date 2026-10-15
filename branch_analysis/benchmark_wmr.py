import glob
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_DIR = os.path.join(HERE, "predictions")
OUT_FILE = os.path.join(HERE, "benchmark_wmr.csv")

rows = []
for path in sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "EMB_CT_*.csv"))):
    bench = os.path.basename(path).replace("EMB_CT_", "").replace(".csv", "")
    df = pd.read_csv(path, low_memory=False)

    # Each branch's weighted error. Weights add up to 1 in a benchmark,
    # so the sum of them * 100 is the benchmark's WMR (%).
    wmr = (1 - df["predicted"]) * df["rate"] * df["weight"] + df["predicted"] * (1 - df["rate"]) * df["weight"]

    # Same error, but with the perfect predictor: every branch gets its label y.
    perfect = (1 - df["y"]) * df["rate"] * df["weight"] + df["y"] * (1 - df["rate"]) * df["weight"]

    rows.append({
        "benchmark": bench,
        "branches": len(df),
        "wmr": 100 * wmr.sum() / df["weight"].sum(),
        "perfect_wmr": 100 * perfect.sum() / df["weight"].sum(),
    })

pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print(f"{len(rows)} benchmarks written to {OUT_FILE}")
