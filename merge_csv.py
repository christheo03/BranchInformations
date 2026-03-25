import pandas as pd

main_csv = pd.read_csv("branches.csv")
results_csv = pd.read_csv("ubd_results.csv")

merged = main_csv.merge(results_csv, on="Address", how="left")

merged.to_csv("branches.csv", index=False)

print("Done. Updated branches.csv")