#!/usr/bin/env python3
# Usage: python enrich_branches.py path/to/binary
# Reads branches.csv (address in first column, hex), writes branches_enriched.csv.

import csv
import logging
import sys

import angr
import networkx as nx

NEW_COLS = [
    "br_is_loop_header",
    "t_dominates",
    "t_post_dominates",
    "t_is_loop_head",
    "t_is_backedge",
    "t_successor_ends",
    "t_is_loop_exit",
    "t_has_call",
    "f_dominates",
    "f_post_dominates",
    "f_is_loop_head",
    "f_is_backedge",
    "f_successor_ends",
    "f_is_loop_exit",
    "f_has_call",
    "branch_bb_addr",
    "taken_bb_addr",
    "fall_bb_addr",
]


def doms_from_idom(g, idom):
    dom = {n: set() for n in g.nodes()}
    for n in g.nodes():
        cur = n
        while cur in idom:
            dom[n].add(cur)
            if idom[cur] == cur:
                break
            cur = idom[cur]
    return dom


def loop_body(g, head, tail):
    body = {head, tail}
    stack = [tail]
    while stack:
        n = stack.pop()
        for p in g.predecessors(n):
            if p not in body:
                body.add(p)
                stack.append(p)
    return body


def analyze_function(func):

    # Create the CFG of function
    g = nx.DiGraph()
    for n in func.graph.nodes():
        if getattr(n, "addr", None) is not None:
            g.add_node(n.addr)
    for u, v in func.graph.edges():
        if getattr(u, "addr", None) is not None and getattr(v, "addr", None) is not None:
            g.add_edge(u.addr, v.addr)
    if func.addr not in g:
        return None
    
    # Creates the dominator tree
    idom = nx.immediate_dominators(g, func.addr)
    dom = doms_from_idom(g, idom)

    r = g.reverse(copy=True)
    VEXIT = "_exit_"
    r.add_node(VEXIT)
    for n in g.nodes():
        if g.out_degree(n) == 0:
            r.add_edge(VEXIT, n)
    if r.out_degree(VEXIT) == 0:
        r.add_edge(VEXIT, func.addr)
    ipdom = nx.immediate_dominators(r, VEXIT)
    pdom = {n: s - {VEXIT} for n, s in doms_from_idom(r, ipdom).items() if n != VEXIT}

    backedges = {(u, v) for u, v in g.edges() if v in dom[u]}
    loop_heads = {v for _, v in backedges}

    bodies = {}
    for tail, head in backedges:
        bodies.setdefault(head, set()).update(loop_body(g, head, tail))
    exits = set()
    for body in bodies.values():
        for u in body:
            for _, w in g.out_edges(u):
                if w not in body:
                    exits.add((u, w))

    return {
        "g": g, "dom": dom, "pdom": pdom,
        "backedges": backedges, "loop_heads": loop_heads, "loop_exits": exits,
    }


def successors(proj, node):
    insns = proj.factory.block(node.addr, size=node.size).capstone.insns
    if not insns:
        return None, None
    last = insns[-1]
    fall_pc = last.address + last.size
    succ = [s.addr for s in node.successors]
    fall = fall_pc if fall_pc in succ else None
    non_fall = [a for a in succ if a != fall_pc]
    taken = non_fall[0] if non_fall else None
    return taken, fall


def cell(v):
    # 1 / 0 for booleans, -1 when unknown.
    return "-1" if v is None else ("1" if v else "0")


def analyze_branch(proj, cfg, addr, cache):
    out = {c: "-1" for c in NEW_COLS}
    node = cfg.model.get_any_node(addr) or cfg.model.get_any_node(addr, anyaddr=True)
    if node is None or node.function_address is None:
        return out
    func = cfg.functions.get(node.function_address)
    if func is None:
        return out

    if func.addr not in cache:
        cache[func.addr] = analyze_function(func)
    A = cache[func.addr]
    if A is None or node.addr not in A["g"]:
        return out

    bb = node.addr
    taken, fall = successors(proj, node)
    g, dom, pdom = A["g"], A["dom"], A["pdom"]

    def known(x):
        return x is not None and x in g

    def dominates(a, b):
        return a in dom[b] if known(a) and known(b) else None

    def postdominates(a, b):
        return a in pdom[b] if known(a) and known(b) else None

    def edge_in(u, v, s):
        if not (known(u) and known(v)):
            return None
        return g.has_edge(u, v) and (u, v) in s

    def last_opcode(x):
        if x is None:
            return "-1"
        n = cfg.model.get_any_node(x)
        if n is None:
            return "-1"
        insns = proj.factory.block(n.addr, size=n.size).capstone.insns
        if not insns:
            return "-1"
        return insns[-1].mnemonic

    def ends_with_call(x):
        op = last_opcode(x)
        if op == "-1":
            return None
        return op == "call"

    out["br_is_loop_header"]  = cell(bb in A["loop_heads"])
    out["t_dominates"]        = cell(dominates(bb, taken))
    out["t_post_dominates"]   = cell(postdominates(taken, bb))
    out["t_is_loop_head"]     = cell(taken in A["loop_heads"]) if known(taken) else "-1"
    out["t_is_backedge"]      = cell(edge_in(bb, taken, A["backedges"]))
    out["t_successor_ends"]   = last_opcode(taken)
    out["t_is_loop_exit"]     = cell(edge_in(bb, taken, A["loop_exits"]))
    out["t_has_call"]         = cell(ends_with_call(taken))
    out["f_dominates"]        = cell(dominates(bb, fall))
    out["f_post_dominates"]   = cell(postdominates(fall, bb))
    out["f_is_loop_head"]     = cell(fall in A["loop_heads"]) if known(fall) else "-1"
    out["f_is_backedge"]      = cell(edge_in(bb, fall, A["backedges"]))
    out["f_successor_ends"]   = last_opcode(fall)
    out["f_is_loop_exit"]     = cell(edge_in(bb, fall, A["loop_exits"]))
    out["f_has_call"]         = cell(ends_with_call(fall))
    out["branch_bb_addr"]     = bb
    out["taken_bb_addr"]      = taken if taken is not None else "-1"
    out["fall_bb_addr"]       = fall  if fall  is not None else "-1"
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: python enrich_branches.py path/to/binary", file=sys.stderr)
        sys.exit(2)
    binary = sys.argv[1]

    for n in ("angr", "cle", "pyvex", "claripy"):
        logging.getLogger(n).setLevel(logging.ERROR)

    proj = angr.Project(binary, auto_load_libs=False)
    cfg = proj.analyses.CFGFast(normalize=True)

    # Load everything first; we'll overwrite branches.csv at the end.
    with open("branches.csv", newline="") as fin:
        rows = list(csv.reader(fin))

    cache = {}
    tmp_path = "branches.csv.tmp"
    with open(tmp_path, "w", newline="") as fout:
        w = csv.writer(fout)
        if len(rows) > 2:
            w.writerow(rows[2] + NEW_COLS)
        for row in rows[3:]:
            addr = int(row[0].strip(), 16)
            extra = analyze_branch(proj, cfg, addr, cache)
            w.writerow(row + [extra[c] for c in NEW_COLS])

    import os
    os.replace(tmp_path, "branches.csv")  # atomic swap
    print("updated branches.csv")


if __name__ == "__main__":
    main()
