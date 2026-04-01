# Christos Theodorou
# Github: christheo03

import json
import csv
import networkx as nx
from bisect import bisect_right
from networkx.drawing.nx_pydot import write_dot

# Input (Graph, node_id, radius, outputfile_name)
# Output produce a .dot file which is a subgraph of the garaph around the node_id 
# def export_dot_subgraph(G, node, radius=1, filename="subgraph.dot"):
#     subG = nx.ego_graph(G, node, radius=radius, undirected=True)

#     # optional: put edge labels in the DOT output
#     for u, v, data in subG.edges(data=True):
#         data["label"] = str(data.get("type_id", ""))

#     write_dot(subG, filename)
#     print(f"Wrote DOT subgraph to {filename}")


BASE = "0x400000"
START = 1
END = 2
CSV_FILENAME ="branches.csv"
DCFG_FILENAME="test.dcfg.json"

# Convert a hexandecimal to decimal
def to_int(pc):
    if isinstance(pc, int): return pc
    return int(pc, 16) if str(pc).startswith('0x') else int(pc)

# Creation of the Graph using the edges
def create_DiGraph(edges_table: list):
    G = nx.DiGraph()
    for row in edges_table[1:]:
        src = row[1]
        dst = row[2]
        edge_type_id = row[3]
        G.add_edge(src, dst, type_id=edge_type_id)
    return G

# Check if node u dominates node v
def dominates(u, v, dom_sets):
    return u in dom_sets.get(v, set())

# Check if v is a loop head
def is_LoopHead(v,loops):
    if v in loops:
        return True
    return False


    # for u in G.predecessors(v):
    #     if dominates(v, u, dom_sets):
    #         return True
    # return False


# Creation of a dictionary with key a node and value its dominators
def create_domsets(idom, Root):
    dom_sets = {}
    for node in idom:
        dom_sets[node] = set()
        cur = node
        while True:
            dom_sets[node].add(cur)
            if cur == Root:
                break
            cur = idom[cur]
    return dom_sets

# Creation of index tables that helps the performance of the serch 
def index_bb(bb_table):
    starts = []
    blocks = []
    for bb in bb_table[1:]:
        node_id=bb[0]
        offset = to_int(bb[1])
        size = bb[2]
        end = offset + size
        starts.append(offset)
        blocks.append((node_id,offset,size,end))
    
    return starts,blocks

# Return the basic block information with input a pc address
def pc_bbid(pc_br, starts, blocks):
    offset = pc_br - to_int(BASE)

    i = bisect_right(starts,offset) - 1

    if i<0:
        return None, None, None
    
    node_id, bb_offset, bb_size, bb_end = blocks[i]

    if offset < bb_end:
        return node_id, bb_offset, bb_size
    
    return None,None,None


def edge_type(bb, G):
    edges = list(G.out_edges(bb, data=True))
    if not edges:
        return -1
    types_ids = []
    for (_, _, data) in edges:
        types_ids.append(data["type_id"])
    for t in types_ids:
        if t != 18:
            return t
    return types_ids[0]


# Input: Routines from json file
# Output: Dictionary of loops with key the loop head node and value the nodes that a part of the body of the loop
def find_loops(routines):
    loops = {}
    for routine in routines[1:]:
        if len(routine) > 3:
            loop_section = routine[3]
            for loop in loop_section[1:]:
                loop_head = loop[0]
                loop_nodes = set(loop[2])
                loops[loop_head] = loop_nodes
    return loops

# Check if an edge from src to dst is a loop exit edge
def is_loop_exit_edge(src_node, dst_node, loops):
    for loop_head, loop_nodes in loops.items():
        if src_node in loop_nodes and dst_node not in loop_nodes:
            return True
    return False

# Check if the edge type of a basic block is a CALL
def has_succ_call(bb, G):
    CALL_TYPES = {3, 4, 5, 7, 19}
    UNCONDITIONAL_BRANCH_TYPES = {10, 15, 16}
    edges = list(G.out_edges(bb, data=True))
    if not edges: return False
    
    try:
        src, dst, data = edges[0]
        edge_type_id = data["type_id"]
        if edge_type_id in CALL_TYPES:
            return True
        if edge_type_id in UNCONDITIONAL_BRANCH_TYPES:
            return has_succ_call(dst, G)
    except (IndexError, RecursionError):
        pass
    return False


# Returns the location of the Main IMAGE that is Address is BASE
def find_img_num(dcfg):
    images =dcfg['PROCESSES'][1][1]['IMAGES']
    for i in range(len(images)):
        if images[i][1]==BASE:
            return i

# Visualise a subgraph arround node
def export_dot_subgraph(G, node, radius=1, filename="subgraph.dot"):
    subG = nx.ego_graph(G, node, radius=radius, undirected=True)
    write_dot(subG, filename)

        
def main():
    # Load information from the json file
    try:
        with open("test.dcfg.json", "r") as f:
            dcfg_json = json.load(f)
            img_number= find_img_num(dcfg_json)
            bb_table = dcfg_json['PROCESSES'][1][1]['IMAGES'][img_number][3]['BASIC_BLOCKS']
            edges_table = dcfg_json['PROCESSES'][1][1]['EDGES']
            routines = dcfg_json['PROCESSES'][1][1]['IMAGES'][img_number][3]['ROUTINES']
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    # Graph analysis information
    loops = find_loops(routines)
    dcfg_graph = create_DiGraph(edges_table)
    rev_graph = dcfg_graph.reverse(copy=False)
    idom = nx.immediate_dominators(dcfg_graph, START)
    ipdom = nx.immediate_dominators(rev_graph, END)
    dom_sets = create_domsets(idom, START)
    ipdom_sets = create_domsets(ipdom, END)
    starts, basic_blocks= index_bb(bb_table)

    updated_rows = []
    # List of inforamtion of each branch
    new_columns = [
        "br_is_loop_header", 
        "t_dominates", "t_post_dominates", "t_is_loop_head", "t_is_backedge", "t_successor_ends", "t_is_loop_exit", "t_has_call", 
        "f_dominates", "f_post_dominates", "f_is_loop_head", "f_is_backedge", "f_successor_ends", "f_is_loop_exit", "f_has_call", 
        "branch_bb_addr", "branch_bb_ends", "taken_bb_addr","taken_bb_ends", "fall_bb_addr", "fall_bb_ends"
    ]

    # 2. Read existing CSV data
    with open(CSV_FILENAME, "r") as f:
        data_lines = [line for line in f if not line.startswith('#')]
        reader = csv.DictReader(data_lines)
        original_fieldnames = reader.fieldnames
        
        for row in reader:
            pc_str = row['Address']
            pc_br = to_int(pc_str)
            br_bbid,bb_br_starts,size_br = pc_bbid(pc_br, starts,basic_blocks)


            analysis = {col: -1 for col in new_columns}

            if br_bbid is not None:
                jump_offset = int(row['Offset'])
                instr_size = int(row['Size'])
                taken_pc = pc_br + jump_offset
                fall_pc = pc_br + instr_size
                
                taken_bbid,bb_t_starts,size_t = pc_bbid(taken_pc, starts,basic_blocks)
                fall_bbid,bb_f_starts,size_f = pc_bbid(fall_pc, starts,basic_blocks)

                analysis["branch_bb_addr"] = bb_br_starts+ int(BASE,16)
                analysis["br_is_loop_header"] = int(is_LoopHead(br_bbid, loops))
                analysis["branch_bb_ends"]= bb_br_starts+size_br+int(BASE,16)

                if taken_bbid is not None:
                    analysis["taken_bb_addr"] = bb_t_starts+ int(BASE,16)
                    analysis["taken_bb_ends"]= bb_t_starts+size_t+int(BASE,16)
                    analysis["t_dominates"] = int(dominates(br_bbid, taken_bbid, dom_sets))
                    analysis["t_post_dominates"] = int(dominates(taken_bbid, br_bbid, ipdom_sets))
                    analysis["t_is_loop_head"] = int(is_LoopHead(taken_bbid, loops))
                    analysis["t_is_backedge"] = int(dominates(taken_bbid, br_bbid, dom_sets))
                    analysis["t_successor_ends"] = edge_type(taken_bbid, dcfg_graph)
                    analysis["t_is_loop_exit"] = int(is_loop_exit_edge(br_bbid, taken_bbid, loops))
                    analysis["t_has_call"] = int(has_succ_call(taken_bbid, dcfg_graph))

                if fall_bbid is not None:
                    analysis["fall_bb_addr"] = bb_f_starts+ int(BASE,16)
                    analysis["fall_bb_ends"]= bb_f_starts+size_f+int(BASE,16)
                    analysis["f_dominates"] = int(dominates(br_bbid, fall_bbid, dom_sets))
                    analysis["f_post_dominates"] = int(dominates(fall_bbid, br_bbid, ipdom_sets))
                    analysis["f_is_loop_head"] = int(is_LoopHead( fall_bbid, loops))
                    analysis["f_is_backedge"] = int(dominates(fall_bbid, br_bbid, dom_sets))
                    analysis["f_successor_ends"] = edge_type(fall_bbid, dcfg_graph)
                    analysis["f_is_loop_exit"] = int(is_loop_exit_edge(br_bbid, fall_bbid, loops))
                    analysis["f_has_call"] = int(has_succ_call(fall_bbid, dcfg_graph))

            row.update(analysis)
            updated_rows.append(row)

    # 3. Write back to CSV
    with open(CSV_FILENAME, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=original_fieldnames + new_columns)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Done! {CSV_FILENAME} updated with all heuristics and successor types.")
    # export_dot_subgraph(dcfg_graph, node=29, radius=3)
if __name__ == "__main__":
    main()
