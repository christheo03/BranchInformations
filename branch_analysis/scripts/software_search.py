import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

PREDICTIONS_DIR = "predictions"

# Returns the total wmr from branches that their group is shared between multiple benchmarks
def wmr_shared(path,spans,wrong=None):
    df =pd.read_csv(path)
    if wrong is True:
        df = df[df["predicted"]!= df["actual"]]
    if wrong is False:
        df = df[df["predicted"]== df["actual"]]
    n = df["Group"].map(spans)
    return df.loc[n>1,"wmr"].sum()

# Returns the number of branches that their group is shared between multiple benchmarks
def static_shared(path,spans,wrong=None):
    df = pd.read_csv(path)
    if wrong is True:
        df = df[df["predicted"]!= df["actual"]]
    if wrong is False:
        df = df[df["predicted"]== df["actual"]]
    n = df["Group"].map(spans)
    return (n>1).sum()

# Returns a map with key group_ID and value total benchmakrs that group appears in
# Returns {int ID : int total_bench} 
def benchmarks_per_group():
    with open("data/identical_br.json") as f:
        groups = json.load(f)
    return {g["group_id"]: len(g["benchmarks"]) for g in groups}

# string to boolean
# Helper function
def str_to_bool(txt):
    if txt.lower() == "true":
        return True
    if txt.lower() == "false":
        return False

# Builds a graph with a given x values and y values.
# Used for a single benchmark
def build_graph(x,y):
    fig, ax = plt.subplots()
    ax.bar(x, y)
    plt.show()

# Builds a graph that shows all the benchmarks wmr 
def build_all_graph(in_groups, no_group, title, what="WMR from", shared=None):
    names = sorted(f.replace("EMB_CT_", "").replace(".csv", "") for f in os.listdir(PREDICTIONS_DIR) if f.startswith("EMB_CT_"))
    fig, ax = plt.subplots()
    if shared is None:
        ax.bar(names, in_groups, color="tab:blue", label=f"{what} Branches in a group")
    else:
        # in_groups already contains shared, so the blue part is what is left
        only_here = [a - b for a, b in zip(in_groups, shared)]
        ax.bar(names, shared, color="tab:green", label=f"{what} Branches in a group that is also in another benchmark")
        ax.bar(names, only_here, bottom=shared, color="tab:blue", label=f"{what} Branches in a group only in their benchmark")
    ax.bar(names, no_group, bottom=in_groups, color="tab:orange", label=f"{what} Branches in no group")

    # Write the number of each colour above its bar, in the order of the legend
    if shared is None:
        parts = [(in_groups, "tab:blue"), (no_group, "tab:orange")]
    else:
        parts = [(shared, "tab:green"), (only_here, "tab:blue"), (no_group, "tab:orange")]
    for i in range(len(names)):
        top = in_groups[i] + no_group[i]
        for k, (values, color) in enumerate(parts):
            value = values[i]
            text = f"{value:.1f}" if isinstance(value, float) else f"{value}"
            ax.annotate(text, (i, top), xytext=(0, 3 + 8 * (len(parts) - 1 - k)), textcoords="offset points",
                        ha="center", fontsize=6, color=color)
    ax.margins(y=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(title)
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()
    
# Spilts the wmr based if branch is in group or not
def split_wmr(path, wrong=None):
    df = pd.read_csv(path)
    if wrong is True:
        df = df[df["predicted"]!= df["actual"]]
    if wrong is False:
        df=df[df["predicted"]== df['actual']]
    in_groups = df.loc[df["Group"] > 0, "wmr"].sum()
    no_group = df.loc[df["Group"] == 0, "wmr"].sum()
    return in_groups,no_group

# Builds one bar for a benchmark, split into the wmr from branches in a group and the wmr from branches in no group
def build_split_graph(name, in_groups, no_group,title):
    fig, ax = plt.subplots()
    ax.bar([name], [in_groups], color="tab:blue", label="WMR from Branches in a group")
    ax.bar([name], [no_group], bottom=[in_groups], color="tab:orange", label="WMR from Branches in no group")
    ax.set_title(title)
    ax.legend()
    plt.show()

# Return the static Branches of a given benchmark 
def split_static(path,wrong=None):
    df = pd.read_csv(path)
    if wrong is True:
        df = df[df["predicted"]!=df["actual"]]
    if wrong is False:
        df = df[df["predicted"] == df["actual"]]
    in_groups = (df["Group"]>0).sum()
    no_group = (df["Group"]==0).sum()
    return in_groups,no_group

# Runs split (split_wmr or split_static) on every benchmark, returns the two lists
def collect(split, wrong):
    in_groups = []
    no_group = []
    for f in sorted(os.listdir(PREDICTIONS_DIR)):
        if f.startswith("EMB_CT"):
            in_g, no_g = split(os.path.join(PREDICTIONS_DIR, f), wrong)
            in_groups.append(in_g)
            no_group.append(no_g)
    return in_groups, no_group

# Graph titles, by the value of --wrong
WMR_TITLES = {
    None: "Mispredicted dynamic branches in each benchmark(%)",
    True: "Mispredicted Dynamic Branches from Mispredicted Static Branches(%)",
    False: "Mispredicted Dynamic Branches from Correct Predicted Static Branches(%)",
}
STATIC_TITLES = {
    None: "Static branches in each benchmark",
    True: "Mispredicted static branches in each benchmark",
    False: "Correctly predicted static branches in each benchmark",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=str, required=False)
    parser.add_argument("--wrong", type=str_to_bool, required=False, default=None)
    parser.add_argument("--graph", type=str_to_bool, required=False)
    parser.add_argument("--static", type=str_to_bool,required=False)
    args = parser.parse_args()
    
    if args.bench:
        path = os.path.join(PREDICTIONS_DIR, f"EMB_CT_{args.bench}.csv")
        in_groups, no_group = split_wmr(path,args.wrong)
        if args.wrong is True:
            title = f"{args.bench}: mispredicted dynamic branches from mispredicted static branches (%)"
        elif args.wrong is False:
            title = f"{args.bench}: mispredicted dynamic branches from correctly predicted static branches (%)"
        else:
            title = f"{args.bench}: mispredicted dynamic branches (%)"
        
        build_split_graph(args.bench, in_groups, no_group,title)

    if args.graph and not args.bench:
        if args.static:
            in_groups, no_group = collect(split_static, args.wrong)
            spans = benchmarks_per_group()
            shared = [static_shared(os.path.join(PREDICTIONS_DIR, f), spans, args.wrong)
                      for f in sorted(os.listdir(PREDICTIONS_DIR)) if f.startswith("EMB_CT")]
            build_all_graph(in_groups, no_group, STATIC_TITLES[args.wrong], what="Static", shared=shared)
        else:
            in_groups, no_group = collect(split_wmr, args.wrong)
            spans = benchmarks_per_group()
            
            shared = [wmr_shared(os.path.join(PREDICTIONS_DIR, f), spans, args.wrong)
                      for f in sorted(os.listdir(PREDICTIONS_DIR)) if f.startswith("EMB_CT")]
            build_all_graph(in_groups, no_group, WMR_TITLES[args.wrong], shared=shared)

if __name__=="__main__":
    main()
