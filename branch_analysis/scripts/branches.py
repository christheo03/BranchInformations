import json
import os
import pandas as pd

PREDICTIONS_DIR = "predictions"

# The paths of the prediction files of all the benchmarks
def benchmark_paths():
    return [os.path.join(PREDICTIONS_DIR, f) for f in sorted(os.listdir(PREDICTIONS_DIR)) if f.startswith("EMB_CT")]

# Returns a map with key group_ID and value total benchmakrs that group appears in
# Returns {int ID : int total_bench}
def benchmarks_per_group():
    with open("data/identical_br.json") as f:
        groups = json.load(f)
    return {g["group_id"]: len(g["benchmarks"]) for g in groups}

# Every static branch of every benchmark, the total the static percentages are based on
def count_static_branches():
    total = 0
    for path in benchmark_paths():
        total += len(pd.read_csv(path, usecols=["Group"]))
    return total

# Reads the branches of a benchmark. wrong=True gives only the mispredicted, wrong=False only the correctly predicted
def read_branches(path, wrong=None):
    df = pd.read_csv(path)
    if wrong is True:
        df = df[df["predicted"] != df["actual"]]
    if wrong is False:
        df = df[df["predicted"] == df["actual"]]
    return df

# From a table of branches, the ones in a group that also has a branch in another benchmark (green)
# spans is the map from benchmarks_per_group()
def with_outside_twin(df, spans):
    return df[df["Group"].map(spans) > 1]

# The ones in a group that exists only in this benchmark (blue)
def group_only_here(df, spans):
    return df[df["Group"].map(spans) == 1]

# The ones in no group (orange)
def no_group_branches(df):
    return df[df["Group"] == 0]

# The wmr of the green, blue and orange branches of a benchmark
def wmr_parts(path, spans, wrong=None):
    df = read_branches(path, wrong)
    return (with_outside_twin(df, spans)["wmr"].sum(),
            group_only_here(df, spans)["wmr"].sum(),
            no_group_branches(df)["wmr"].sum())

# How many green, blue and orange branches are in a table
def count_parts(df, spans):
    return len(with_outside_twin(df, spans)), len(group_only_here(df, spans)), len(no_group_branches(df))

# The number of green, blue and orange branches of a benchmark that contribute to the wmr
def static_parts(path, spans, wrong=None):
    df = read_branches(path, wrong)
    return count_parts(df[df["wmr"] > 0], spans)

# The number of green, blue and orange branches that were predicted right, even the ones that add nothing to the wmr
# (wrong is not used, it is here so it can be used like the other two)
def correct_parts(path, spans, wrong=None):
    return count_parts(read_branches(path, False), spans)

# Runs parts (wmr_parts, static_parts or correct_parts) on every benchmark, returns the three lists
def collect(parts, spans, wrong):
    green = []
    blue = []
    orange = []
    for path in benchmark_paths():
        g, b, o = parts(path, spans, wrong)
        green.append(g)
        blue.append(b)
        orange.append(o)
    return green, blue, orange
