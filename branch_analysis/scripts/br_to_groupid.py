"""
Make a csv that says, for every branch that has an identical-feature twin,
which group it is in: benchmark, pc, group.

Reads data/identical_br.json (made by identical_br.py), so the group
numbers always match it. Run identical_br.py first if you retrained.

Run from branch_analysis/:
    python scripts/br_to_groupid.py
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # branch_analysis/
JSON_FILE = os.path.join(ROOT, "data", "identical_br.json")
OUT_FILE = os.path.join(ROOT, "data", "br_to_groupid.csv")

with open(JSON_FILE) as f:
    groups = json.load(f)

rows = []
for g in groups:
    for bench, info in g["benchmarks"].items():
        for pc in info["pcs"]:
            rows.append({"benchmark": bench, "pc": pc, "group": g["group_id"]})

pd.DataFrame(rows).to_csv(OUT_FILE, index=False)
print(f"{len(rows)} branches in {len(groups)} groups written to {OUT_FILE}")
