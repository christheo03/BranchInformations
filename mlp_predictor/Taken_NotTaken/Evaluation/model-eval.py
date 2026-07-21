import os
import sys
import pandas as pd
import torch

# Run from mlp_predictor/ as the working directory (same assumption
# MODELS_DIR below makes) - `python script.py` only adds the script's own
# folder to the import path, not the cwd, so mlp_model/data_engin/ml_configs
# (which live in mlp_predictor/) wouldn't be found otherwise.
sys.path.append(".")

from helper import *
from data_engin import load_data, add_register_features, add_label_tnt
from ml_configs import (
    TEST_FILES, DATA_DIR,CT_COLUMNS, ESP_COLUMNS
)

MODELS_DIR = "./Taken_NotTaken/Models"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Benchmarks used for evaluation ({len(TEST_FILES)}): {TEST_FILES}")
    print()

    # Tag each row with which benchmark it came from before concatenating -
    # load_data itself just stacks the CSVs together with no such column.
    benchmark_dfs = []
    for name in TEST_FILES:
        d = load_data(DATA_DIR, [name])
        d["Benchmark"] = name
        benchmark_dfs.append(d)
    test_df = pd.concat(benchmark_dfs, ignore_index=True)

    test_df["y"] = add_label_tnt(test_df)
    weights = torch.tensor(test_df["weight"].values, dtype=torch.float32)
    rates = torch.tensor(test_df["rate"].values, dtype=torch.float32)
    y = torch.tensor(test_df["y"].values, dtype=torch.float32)
    executed = torch.tensor(test_df["Executed"].values, dtype=torch.float32)

    for filename in sorted(os.listdir(MODELS_DIR)):
        if filename not in ("SS_CT.pth", "ESP.pth", "EMB_CT.pth"):
            continue

        checkpoint = torch.load(f"{MODELS_DIR}/{filename}", weights_only=False)

        df = test_df.copy()
        add_register_features(df, df)

        if filename == "SS_CT.pth":
            model = build_ss_ct(checkpoint).to(device)
            X = normalize_ss(df, checkpoint, CT_COLUMNS).to(device)
            with torch.no_grad():
                outputs = model(X)

        elif filename == "ESP.pth":
            model = build_esp(checkpoint).to(device)
            X = normalize_ss(df, checkpoint, ESP_COLUMNS).to(device)
            with torch.no_grad():
                outputs = model(X)

        elif filename == "EMB_CT.pth":
            model = build_emb_ct(checkpoint).to(device)
            X_num, X_cat = normalize_emb(df, checkpoint, CT_COLUMNS)
            X_num, X_cat = X_num.to(device), X_cat.to(device)
            with torch.no_grad():
                outputs = model(X_num, X_cat)

        outputs = outputs.cpu()
        results = evaluate(outputs, y, weights, rates, executed)
        confusion = results["confusion"]

        print(f"=== {filename} ===")
        print(f"Weighted Miss Rate: {results['weighted_miss_rate'] * 100:.3f}%")
        print(f"Dynamic Miss Rate:  {results['dynamic_miss_rate'] * 100:.3f}%")
        print(f"Static Miss Rate:   {results['static_miss_rate'] * 100:.3f}%")
        print("Confusion Table:")
        print(f"                 Actual Taken   Actual Not-Taken")
        print(f"  Pred Taken     {confusion['TP']:<14} {confusion['FP']}")
        print(f"  Pred Not-Taken {confusion['FN']:<14} {confusion['TN']}")
        print()

        mispred = mispredicted_branches(outputs, y, test_df)
        cols = [c for c in ["Address", "Benchmark"] if c in mispred.columns]
        print(f"Top 5 Mispredicted Branches (of {len(mispred)}), sorted by Executed:")
        print(mispred[cols].head(5).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
