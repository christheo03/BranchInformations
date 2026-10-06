import pandas as pd
import numpy as np
import os


def main():
    dir = "./"
    rows = []

    for file in sorted(os.listdir(dir)):
        if not file.endswith(".csv"):
            continue

        df = pd.read_csv(file)

        weight = df["weight"].values
        rate = df["rate"].values

        # get_binary_error compares the thresholded prediction against the
        # continuous rate, so even a perfectly-directed guess still eats
        # (1 - rate) worth of executions that went the other way - this
        # floor is genuinely non-zero.
        weighted_floor = float(np.sum(weight * np.minimum(rate, 1.0 - rate)))

        rows.append((file, weighted_floor))

        print(f"{file:<25} weighted_floor: {weighted_floor * 100:7.3f}%")

    avg_weighted = sum(r[1] for r in rows) / len(rows)

    print(f"\n{'#' * 50}\nAVERAGE ACROSS {len(rows)} BENCHMARKS\n{'#' * 50}")
    print(f"  weighted_floor: {avg_weighted * 100:7.3f}%")
    print(
        "\n(1-accuracy has no meaningful floor to compute here - it compares "
        "against the discrete label (rate > 0.5), so a rate-aware predictor "
        "always lands on the right side of 0.5 for every branch by "
        "construction, giving a floor of exactly 0% regardless of data.)"
    )


if __name__ == "__main__":
    main()
