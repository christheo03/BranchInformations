"""
Find branches (across all benchmarks) that have exactly the same EMB_CT
input features, and write them to duplicate_feature_summary.json.

Features are rebuilt the same way build_features does it, but kept raw
(before StandardScaler / vocab encoding - both are 1-to-1, so equality is
unchanged). Branches in the same group are indistinguishable to EMB_CT.

Run from mlp_predictor/:
    python find_duplicate_features.py
"""
import json
import os

from ml_configs import pd, ALL_FILES
from ml_configs import CONT_COLS, BIN_COLS, ROUT_COL, REG_COLS, OPC_COLS, ROUTINE_TYPE_MAP
from data_engin import add_register_features, add_label_tnt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "../results")
OUT_FILE = os.path.join(HERE, "duplicate_feature_summary.json")
FEATURE_COLS = CONT_COLS + BIN_COLS + [ROUT_COL] + REG_COLS + OPC_COLS


def load_features(name):
    df = pd.read_csv(os.path.join(DATA_DIR, name + ".csv"), low_memory=False)
    df["file"] = name
    df["y"] = add_label_tnt(df)

    add_register_features(df, df)

    for col in CONT_COLS + BIN_COLS:
        df[col] = df[col].fillna(0)
    df[ROUT_COL] = df[ROUT_COL].map(ROUTINE_TYPE_MAP).fillna(0).astype(int)
    for col in REG_COLS + OPC_COLS:
        df[col] = df[col].astype(str).replace("nan", "-1")

    return df


def to_python(v):
    return v.item() if hasattr(v, "item") else v


def main():
    df = pd.concat([load_features(f) for f in ALL_FILES], ignore_index=True)

    df["group_size"] = df.groupby(FEATURE_COLS)["Address"].transform("size")
    dups = df[df["group_size"] > 1]

    # Largest group first, numbered from 1
    groups = sorted(dups.groupby(FEATURE_COLS, sort=False), key=lambda kv: -len(kv[1]))

    out = []
    for gid, (_, g) in enumerate(groups, start=1):
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

    print(f"{len(dups)} of {len(df)} branches share features with another branch")
    print(f"{len(out)} groups written to {OUT_FILE}")


if __name__ == "__main__":
    main()
