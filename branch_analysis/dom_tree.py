# Given a binary and a basic block address, prints the X closest dominators of that block
# (immediate dominator, then its immediate dominator, ... up to the function entry).
# Uses the same CFG construction as angrversion.py, so block addresses match branch_bb_addr.

import argparse
import logging

import angr
import networkx as nx


def function_graph(func):
    g = nx.DiGraph()
    for n in func.graph.nodes():
        if getattr(n, "addr", None) is not None:
            g.add_node(n.addr)
    for u, v in func.graph.edges():
        if getattr(u, "addr", None) is not None and getattr(v, "addr", None) is not None:
            g.add_edge(u.addr, v.addr)
    return g


def describe(proj, cfg, addr):
    node = cfg.model.get_any_node(addr)
    if node is None or not node.size:
        return f"{addr:#x}"
    insns = proj.factory.block(node.addr, size=node.size).capstone.insns
    if not insns:
        return f"{addr:#x}  size={node.size}"
    last = insns[-1]
    return f"{addr:#x}  size={node.size:<4} last: {last.mnemonic} {last.op_str}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("binary", help="path to the binary (the one the CFG is built from)")
    ap.add_argument("address", help="basic block address, e.g. 0x4e9b90")
    ap.add_argument("-x", "--depth", type=int, default=5,
                    help="how many dominators to print (default 5)")
    args = ap.parse_args()

    for n in ("angr", "cle", "pyvex", "claripy"):
        logging.getLogger(n).setLevel(logging.ERROR)

    addr = int(args.address, 16)
    proj = angr.Project(args.binary, auto_load_libs=False)
    cfg = proj.analyses.CFGFast(normalize=True)

    # the address may be an instruction inside a block (e.g. the branch itself), so use anyaddr
    node = cfg.model.get_any_node(addr) or cfg.model.get_any_node(addr, anyaddr=True)
    if node is None or node.function_address is None:
        raise SystemExit(f"no basic block found containing {addr:#x}")
    func = cfg.functions.get(node.function_address)
    if func is None:
        raise SystemExit(f"no function found for {addr:#x}")

    g = function_graph(func)
    if func.addr not in g or node.addr not in g:
        raise SystemExit(f"block {node.addr:#x} is not in the CFG of {func.name}")

    idom = nx.immediate_dominators(g, func.addr)
    if node.addr not in idom:
        raise SystemExit(f"block {node.addr:#x} is unreachable from the entry of {func.name} in this CFG")

    print(f"function: {func.name} @ {func.addr:#x}")
    print(f"block:    {describe(proj, cfg, node.addr)}")
    cur = node.addr
    for i in range(1, args.depth + 1):
        parent = idom[cur]
        if parent == cur:
            print(f"-- {cur:#x} is the function entry, no more dominators --")
            break
        print(f"idom^{i}:   {describe(proj, cfg, parent)}")
        cur = parent


if __name__ == "__main__":
    main()
