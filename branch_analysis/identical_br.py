"""
Find branches (across all benchmarks) that have exactly the same EMB_CT
input features, and write them to identical_br.json, plus a one-line-per-group
summary to identical_br.csv.

Features are rebuilt the same way build_features does it, but kept raw
(before StandardScaler / vocab encoding - both are 1-to-1, so equality is
unchanged). Branches in the same group are indistinguishable to EMB_CT.

Run from branch_analysis/:
    PYTHONPATH=../mlp_predictor python identical_br.py
"""
import json
import os

from ml_configs import pd, ALL_FILES
from ml_configs import CONT_COLS, BIN_COLS, ROUT_COL, REG_COLS, OPC_COLS, ROUTINE_TYPE_MAP
from data_engin import add_register_features

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "predictions")
OUT_FILE = os.path.join(HERE, "identical_br.json")
SUMMARY_FILE = os.path.join(HERE, "identical_br.csv")
FEATURE_COLS = CONT_COLS + BIN_COLS + [ROUT_COL] + REG_COLS + OPC_COLS


def load_features(name):
    df = pd.read_csv(os.path.join(DATA_DIR, "EMB_CT_" + name + ".csv"), low_memory=False)
    df["file"] = name

    # WMR of each single branch (its weighted error)
    df["wmr"] = (1 - df["predicted"]) * df["rate"] * df["weight"] + df["predicted"] * (1 - df["rate"]) * df["weight"]

    # WMR of each branch if it had been predicted wrong (the opposite of its label y)
    wrong = 1 - df["y"]
    df["wmr_if_wrong"] = (1 - wrong) * df["rate"] * df["weight"] + wrong * (1 - df["rate"]) * df["weight"]

    add_register_features(df, df)
    df[ROUT_COL] = df[ROUT_COL].map(ROUTINE_TYPE_MAP).fillna(0).astype(int)

    return df


# How many branches are in a group (used to sort groups, biggest first).
def group_size(group):
    features, branches = group
    return len(branches)


def to_python(v):
    return v.item() if hasattr(v, "item") else v


def main():
    df = pd.concat([load_features(f) for f in ALL_FILES], ignore_index=True)

    df["group_size"] = df.groupby(FEATURE_COLS)["Address"].transform("size")
    dups = df[df["group_size"] > 1]

    # Largest group first, numbered from 1
    groups = list(dups.groupby(FEATURE_COLS, sort=False))
    groups.sort(key=group_size, reverse=True)

    out = []
    summary = []
    for gid, (_, g) in enumerate(groups, start=1):
        # Weights add up to 1 inside one benchmark, so 100 * wmr is the
        # group's share (%) of that benchmark's weighted miss rate.
        # 0 when the group has no branches from that benchmark.
        wmr_by_bench = g.groupby("file")["wmr"].sum()
        bench_wmr = {f"wmr_{bench}": 100 * wmr_by_bench.get(bench, 0.0) for bench in ALL_FILES}

        summary.append({
            "group": gid,
            "taken": int((g["y"] == 1).sum()),
            "not_taken": int((g["y"] == 0).sum()),
            "dynamic_taken": int(g["Taken"].sum()),
            "dynamic_not_taken": int((g["Executed"] - g["Taken"]).sum()),
            "predicted_correct": int((g["predicted"] == g["y"]).sum()),
            "predicted_wrong": int((g["predicted"] != g["y"]).sum()),
            "wmr_sum": sum(bench_wmr.values()),
            "wmr_sum_if_wrong": 100 * g["wmr_if_wrong"].sum(),
            **bench_wmr,
        })
        out.append({
            "group_id": gid,
            "features": {c: to_python(g.iloc[0][c]) for c in FEATURE_COLS},
            "total_branches": len(g),
            "taken": int((g["y"] == 1).sum()),
            "not_taken": int((g["y"] == 0).sum()),
            "benchmarks": {
                bench: {
                    "branches": len(b),
                    "taken": int((b["y"] == 1).sum()),
                    "not_taken": int((b["y"] == 0).sum()),
                    "pcs": b["Address"].tolist(),
                }
                for bench, b in g.groupby("file", sort=True)
            },
        })

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)

    pd.DataFrame(summary).sort_values("wmr_sum", ascending=False).to_csv(SUMMARY_FILE, index=False)

    print(f"{len(dups)} of {len(df)} branches share features with another branch")
    print(f"{len(out)} groups written to {OUT_FILE}")
    print(f"group summary written to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
