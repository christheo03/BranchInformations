import matplotlib.pyplot as plt
import os
from branches import benchmark_paths

# Graph titles, by the value of --wrong
WMR_TITLES = {
    None: "Mispredicted dynamic branches in each benchmark(%)",
    True: "Mispredicted Dynamic Branches from Mispredicted Static Branches(%)",
    False: "Mispredicted Dynamic Branches from Correct Predicted Static Branches(%)",
}
# Graph titles and bar names for --bench
BENCH_TITLE = "mispredicted dynamic branches (%)"
BENCH_STATIC_TITLE = "static branches that contribute to the WMR"
BENCH_LABELS = {
    None: "all branches",
    False: "correctly predicted\nstatic branches",
    True: "mispredicted\nstatic branches",
}
STATIC_TITLES = {
    None: "Static branches that contribute to the WMR in each benchmark",
    True: "Mispredicted static branches that contribute to the WMR in each benchmark",
    False: "Correctly predicted static branches that contribute to the WMR in each benchmark",
}

# Builds a graph that shows all the benchmarks, sorted by the green part
def build_all_graph(shared, only_here, no_group, title, what="WMR from", denominator=None):
    names = [os.path.basename(p).replace("EMB_CT_", "").replace(".csv", "") for p in benchmark_paths()]
    # Sort the benchmarks by the green part, biggest first
    order = sorted(range(len(names)), key=lambda i: shared[i], reverse=True)
    names = [names[i] for i in order]
    shared = [shared[i] for i in order]
    only_here = [only_here[i] for i in order]
    no_group = [no_group[i] for i in order]
    in_groups = [a + b for a, b in zip(shared, only_here)]

    # Only the static graphs report percentages, based on every static branch of all the benchmarks
    total = sum(in_groups) + sum(no_group)
    if denominator is None:
        denominator = total
    if what == "Static":
        title = title + f"\n{total} of the {denominator} static branches of all the benchmarks ({100 * total / denominator:.1f}%)"
    def legend_text(text, values):
        if what == "Static":
            return f"{text} ({100 * sum(values) / denominator:.1f}%)"
        return text

    fig, ax = plt.subplots()
    ax.bar(names, shared, color="tab:green", label=legend_text(f"{what} Branches in a group that is also in another benchmark", shared))
    ax.bar(names, only_here, bottom=shared, color="tab:blue", label=legend_text(f"{what} Branches in a group only in their benchmark", only_here))
    ax.bar(names, no_group, bottom=in_groups, color="tab:orange", label=legend_text(f"{what} Branches in no group", no_group))

    # Write the number of each colour above its bar, in the order of the legend
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

# One graph with the bars side by side, all on the same y axis. Each one is (label, green, blue, orange)
def build_split_graph(name, graphs, what, title):
    labels = [g[0] for g in graphs]
    shared = [g[1] for g in graphs]
    only_here = [g[2] for g in graphs]
    no_group = [g[3] for g in graphs]
    in_groups = [a + b for a, b in zip(shared, only_here)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, shared, color="tab:green", label=f"{what} Branches in a group that is also in another benchmark")
    ax.bar(labels, only_here, bottom=shared, color="tab:blue", label=f"{what} Branches in a group only in their benchmark")
    ax.bar(labels, no_group, bottom=in_groups, color="tab:orange", label=f"{what} Branches in no group")

    # Write the number of each colour above its bar, in the order of the legend
    parts = [(shared, "tab:green"), (only_here, "tab:blue"), (no_group, "tab:orange")]
    for i in range(len(labels)):
        top = in_groups[i] + no_group[i]
        for k, (values, color) in enumerate(parts):
            value = values[i]
            text = f"{value:.2f}" if isinstance(value, float) else f"{value}"
            ax.annotate(text, (i, top), xytext=(0, 3 + 12 * (len(parts) - 1 - k)), textcoords="offset points",
                        ha="center", fontsize=10, color=color)
    ax.margins(y=0.4)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()
