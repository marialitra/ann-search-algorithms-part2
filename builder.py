import argparse
import os
import re
from typing import Dict, List, Tuple
from collections import Counter as counter

import numpy as np
import kahip


def parse_neighbor_file(path: str) -> Tuple[Dict[int, List[int]], int]:
    """Parse the neighbor file.

    Returns:
        neighbors: dict query -> list of neighbor ids (ints) in order
    """
    neighbors: Dict[int, List[int]] = {}

    # regex helpers
    q_re = re.compile(r"^Query:\s*(\d+)")
    nn_re = re.compile(r"^Nearest neighbor-\d+:\s*(\d+)")

    current_q = None
    # temporary index to align distances
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = q_re.match(line)
            if m:
                current_q = int(m.group(1))
                neighbors[current_q] = []
                continue

            m = nn_re.match(line)
            if m and current_q is not None:
                nid = int(m.group(1))
                neighbors[current_q].append(nid)
                continue

    return neighbors, len(neighbors)


def build_csr_from_neighbors(neighbors: Dict[int, List[int]], datasetsize: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # build edge weight map
    array = {}
    for qid, nlist in neighbors.items():
        for nid in nlist:
            if qid == nid:
                continue
            array[(qid, nid)] = array.get((qid, nid), 0) + 1
            array[(nid, qid)] = array.get((nid, qid), 0) + 1
    sorted_values = dict(sorted(array.items(), key=lambda item: item[0][0]))
    print(sorted_values)
    out_cols = []
    out_data = []

    for r, c in sorted_values.keys():
        out_cols.append(c)
        out_data.append(sorted_values[r, c])


    rows = [k[0] for k in array.keys()]
    counts = counter(rows)

    xadj = np.zeros(datasetsize + 1, dtype=np.int32)

    cum = 0
    for i in range(datasetsize):
        xadj[i] = cum 
        cum += counts.get(i, 0)
    xadj[datasetsize] = cum

    # print("xadj:")
    # print(xadj)

    adjncy = np.array(out_cols, dtype=np.int32)
    # print("adjncy")
    # print(adjncy)
    adjwgt = np.array(out_data, dtype=np.int32)
    # print("adjwgt")
    # print(adjwgt)
    vwgt = np.ones(datasetsize, dtype=np.int32)
    # print("vwgt")
    # print(vwgt)
    return xadj, adjncy, adjwgt, vwgt


def main():
    p = argparse.ArgumentParser(description="Build adjacency matrix (CSR) from neighbor TXT files.")
    p.add_argument("input", help="path to neighbor TXT file")
    args = p.parse_args()

    inp = args.input
    if not os.path.exists(inp):
        raise SystemExit(f"File not found: {inp}")

    neighbors, datasetsize = parse_neighbor_file(inp)
    print(f"Parsed {datasetsize} queries. Building adjacency...")

    xadj, adjncy, adjwgt, vwgt = build_csr_from_neighbors(neighbors, datasetsize)

    N_BLOCKS = 10
    IMBALANCE = 0.03
    SEED = 42

    # Call kahip partitioner
    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjwgt, adjncy, N_BLOCKS, IMBALANCE, True, SEED, 1)
    print(f"Partitioned graph into {N_BLOCKS} blocks with edgecut {edgecut}.")
    print(blocks)
    


if __name__ == "__main__":
    main()
