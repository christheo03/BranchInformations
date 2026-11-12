import argparse
import os
import pandas as pd
from branches import (PREDICTIONS_DIR, benchmarks_per_group, count_static_branches, read_branches,
                      with_outside_twin, group_only_here, wmr_parts, static_parts, correct_parts, collect)
from graphs import (build_all_graph, build_split_graph, WMR_TITLES, STATIC_TITLES,
                    BENCH_TITLE, BENCH_STATIC_TITLE, BENCH_LABELS)

# string to boolean
# Helper function
def str_to_bool(txt):
    if txt.lower() == "true":
        return True
    if txt.lower() == "false":
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch",type=str,required=False)
    parser.add_argument("--bench", type=str, required=False)
    parser.add_argument("--wrong", type=str_to_bool, required=False, default=None)
    parser.add_argument("--graph", type=str_to_bool, required=False)
    parser.add_argument("--static", type=str_to_bool,required=False)
    parser.add_argument("--correct", type=str_to_bool, required=False)
    parser.add_argument("--group", type=int, required=False)
    args = parser.parse_args()

    spans = benchmarks_per_group()

    # --correct True: all the correctly predicted static branches, one graph (for --bench or for all the benchmarks)
    if args.correct:
        if args.bench:
            path = os.path.join(PREDICTIONS_DIR, f"EMB_CT_{args.bench}.csv")
            green, blue, orange = correct_parts(path, spans)
            build_split_graph(args.bench, [(args.bench, green, blue, orange)], "Static",
                              f"{args.bench}: all correctly predicted static branches")
        else:
            shared, only_here, no_group = collect(correct_parts, spans, None)
            build_all_graph(shared, only_here, no_group, "All correctly predicted static branches in each benchmark",
                            "Static", count_static_branches())
        return

    if args.bench and not args.branch:
        path = os.path.join(PREDICTIONS_DIR, f"EMB_CT_{args.bench}.csv")
        # Three bars: all branches, only the correctly predicted, only the mispredicted
        # --static counts the static branches that contribute to the wmr, otherwise it shows the wmr
        if args.static:
            parts, title, what = static_parts, BENCH_STATIC_TITLE, "Static"
        else:
            parts, title, what = wmr_parts, BENCH_TITLE, "WMR from"
        graphs = []
        for wrong in (None, False, True):
            green, blue, orange = parts(path, spans, wrong)
            graphs.append((BENCH_LABELS[wrong], green, blue, orange))
        build_split_graph(args.bench, graphs, what, f"{args.bench}: {title}")

    if args.graph and not args.bench:
        # --static counts the static branches that contribute to the wmr, otherwise it shows the wmr
        if args.static:
            parts, title, what, denominator = static_parts, STATIC_TITLES[args.wrong], "Static", count_static_branches()
        else:
            parts, title, what, denominator = wmr_parts, WMR_TITLES[args.wrong], "WMR from", None
        shared, only_here, no_group = collect(parts, spans, args.wrong)
        build_all_graph(shared, only_here, no_group, title, what, denominator)

    if args.group:
        branches = []
        for f in sorted(os.listdir("predictions")):
            if f.startswith("EMB_CT"):
                df = pd.read_csv("predictions/" + f, dtype=str)
                df = df[df["Group"]==str(args.group)]
                df.insert(0,"bench",f.replace("EMB_CT","").replace(".csv", ""))
                branches.append(df)
        pd.concat(branches).to_csv("data/group_"+str(args.group) + ".csv",index=False)

    # --branch ADDRESS --bench BENCH: in which team is that branch, twins outside the benchmark, twins only inside it or no twins
    if args.branch and args.bench:
        df = read_branches(f"{PREDICTIONS_DIR}/EMB_CT_{args.bench}.csv")
        found = df[df["Address"].str.lower() == args.branch.lower()]
        name = f"Branch {args.branch} of {args.bench}"
        if found.empty:
            print(f"There is no branch {args.branch} in {args.bench}")
        elif len(with_outside_twin(found, spans)) > 0:
            group = found.iloc[0]["Group"]
            print(f"{name} has twins outside the benchmark: group {group}, which is in {spans[group]} benchmarks")
        elif len(group_only_here(found, spans)) > 0:
            group = found.iloc[0]["Group"]
            print(f"{name} has twins only inside the benchmark: group {group}, with {(df['Group'] == group).sum() - 1} other branches")
        else:
            print(f"{name} has no twins, it is in no group")

if __name__=="__main__":
    main()
