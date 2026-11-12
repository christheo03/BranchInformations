import json
import pandas as pd

BENCH = "520.omnetpp_r"

path = "predictions/EMB_CT_" + BENCH + ".csv"
raw = pd.read_csv(path, dtype=str, keep_default_na=False)  # as they are in the file
df = pd.read_csv(path)  # with numbers, to filter and sort

with open("data/identical_br.json") as f:
    groups = json.load(f)

# How many benchmarks each group appears in
benchmarks = {}
for g in groups:
    benchmarks[g["group_id"]] = len(g["benchmarks"])

# The branches predicted wrong whose group appears in more than one benchmark
n = df["Group"].map(benchmarks)
keep = (n > 1) & (df["predicted"] != df["actual"])

# Sorted from the one that contributes the most to the least
order = df.loc[keep, "wmr"].sort_values(ascending=False).index
raw.loc[order].to_csv("data/shared_wrong_" + BENCH + ".csv", index=False)
