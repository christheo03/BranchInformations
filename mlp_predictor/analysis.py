import glob
import os

import pandas as pd

PREDICTIONS_DIR = "./predictions" # Directory of the predictions

QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]

# Branches inside this quantile (e.g. 0.95 = Q-95) are exported to a csv.
EXPORT_QUANTILE = 0.5
EXPORT_PATH = "./q_branches.csv"

# Per-branch weighted error (get_binary_error)
def row_errors(predicts):
    return (1 - predicts["predicted"]) * predicts["rate"] * predicts["weight"] + predicts["predicted"] * (1 - predicts["rate"]) * predicts["weight"]

# Returns the wmr from a given serie of branches
# Full weight is always ~1
def wmr_series(predicts,full_weight):
    errors = row_errors(predicts)
    wmr = 100 * errors.sum() / full_weight if full_weight else 0.0
    return wmr

# For each quantile in QUANTILES, how many branches (ranked by their own
# weighted-error contribution, biggest first) are needed until they add up
# to that fraction of the serie's total error - same idea as Table VI's
# Q-50/Q-75/... but applied to WMR instead of execution count.
def wmr_quantiles(predicts):
    errors = row_errors(predicts).sort_values(ascending=False)
    total_error = errors.sum()
    cumulative = errors.cumsum()

    result = []
    for q in QUANTILES:
        n = min((cumulative < q * total_error).sum() + 1, len(errors)) if total_error else 0
        result.append(n)
    return result

# The actual branches (all features + predicted/actual) that make up the
# given quantile, biggest wmr contributor first.
def quantile_branches(predicts, quantile):
    errors = row_errors(predicts).sort_values(ascending=False)
    total_error = errors.sum()
    n = min((errors.cumsum() < quantile * total_error).sum() + 1, len(errors)) if total_error else 0

    branches = predicts.loc[errors.index[:n]].copy()
    branches["wmr_contribution"] = errors.iloc[:n].values
    branches["rank"] = range(1, n + 1)
    return branches

# Returns the total number of hb branches based on bias in a serie
# Percentage of hb_total in the serie
def hb(serie,bias,total):
    hb_total = ((serie["rate"]<=(1-bias)) | (serie["rate"]>=bias)).sum()
    hb_pct = 100 * hb_total / total

    return hb_total,hb_pct


def main():
    exported = []

    print(f"{'Benchmark':<20}{'Total':>10}{'HB95':>10}{'HB95 %':>10}{'HB99':>10}{'HB99 %':>10}"
      f"{'Misclass':>10}{'Misclass %':>12}{'Wrong WMR %':>13}{'Correct WMR %':>15}"
      f"{'Wrong HB99':>12}{'Wrong HB99 %':>14}"
      + "".join(f"{'Q-' + str(int(q*100)):>8}" for q in QUANTILES))

    for path in sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.csv"))):
        bench = os.path.basename(path).replace("EMB_CT_", "").replace(".csv", "")
        df = pd.read_csv(path)

        hb95, hb95_pct = hb(df,0.95,len(df))
        hb99, hb99_pct = hb(df,0.99,len(df))

        wrong = df[df["predicted"] != df["actual"]] # Misclassified branches
        correct = df[df["predicted"] == df["actual"]] # Correct prediction
        misclassified = len(wrong)

        wrong_hb99,wrong_hb99_pct = hb(wrong,0.99,misclassified)

        wrong_q = wmr_quantiles(wrong)

        q_branches = quantile_branches(wrong, EXPORT_QUANTILE)
        q_branches.insert(0, "Benchmark", bench)
        exported.append(q_branches)

        full_weight = df["weight"].sum()

        wrong_wmr = wmr_series(wrong,full_weight) # WMR from misclassifications
        correct_wmr = wmr_series(correct,full_weight) # WMR from correct classified
        misclassified_pct = 100 * misclassified / len(df)

        print(f"{bench:<20}{len(df):>10}{hb95:>10}{hb95_pct:>9.2f}%{hb99:>10}{hb99_pct:>9.2f}%"
            f"{misclassified:>10}{misclassified_pct:>11.2f}%{wrong_wmr:>12.2f}%{correct_wmr:>14.2f}%"
            f"{wrong_hb99:>12}{wrong_hb99_pct:>13.2f}%"
            + "".join(f"{n:>8}" for n in wrong_q))

    pd.concat(exported, ignore_index=True).to_csv(EXPORT_PATH, index=False)
    print(f"\nQ-{int(EXPORT_QUANTILE*100)} branches -> {EXPORT_PATH}")





if __name__ == "__main__":
    main()
