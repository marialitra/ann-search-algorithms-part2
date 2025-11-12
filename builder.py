#!/usr/bin/env python3
"""Parse neighbor graph text files (the "Query:/Nearest neighbor-#/distanceApproximate" style)
and build an adjacency matrix in CSR format WITHOUT SciPy.

Outputs (saved next to input file):
 - <input>.indptr.npy
 - <input>.indices.npy
 - <input>.data.npy

This script always produces the three CSR arrays using NumPy. By default the
weighting rule is "mutual": an undirected adjacency where an edge weight is
1.0 if only one direction exists (a->b or b->a) and 2.0 if both list each
other. Use `--distance` to instead use the numeric distance values found in
the file (default behavior previously).
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import kahip


def parse_neighbor_file(path: str) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    """Parse the neighbor file.

    Returns:
        neighbors: dict query -> list of neighbor ids (ints) in order
        weights: dict query -> list of weights (floats) aligned with neighbors
    """
    neighbors: Dict[int, List[int]] = {}
    weights: Dict[int, List[float]] = {}

    # regex helpers
    q_re = re.compile(r"^Query:\s*(\d+)")
    nn_re = re.compile(r"^Nearest neighbor-\d+:\s*(\d+)")
    d_re = re.compile(r"^distanceApproximate:\s*([0-9.+-eE]+)")

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
                weights[current_q] = []
                continue

            m = nn_re.match(line)
            if m and current_q is not None:
                nid = int(m.group(1))
                neighbors[current_q].append(nid)
                # placeholder for the distance value which should follow
                weights[current_q].append(np.nan)
                continue

            m = d_re.match(line)
            if m and current_q is not None:
                val = float(m.group(1))
                # assign to last appended weight
                if neighbors[current_q]:
                    weights[current_q][-1] = val
                continue

    return neighbors, weights


def build_csr_from_neighbors(neighbors: Dict[int, List[int]], weights: Dict[int, List[float]],
                                symmetric: bool = False):
    # compute node count
    max_node = -1
    for q, nbrs in neighbors.items():
        if q > max_node:
            max_node = q
        if nbrs:
            m = max(nbrs)
            if m > max_node:
                max_node = m
    n = max_node + 1

    rows = []
    cols = []
    data = []
    for q, nbrs in neighbors.items():
        ws = weights.get(q, [])
        for i, nb in enumerate(nbrs):
            rows.append(q)
            cols.append(nb)
            # if weight is nan (missing), default to 1.0; otherwise use the distance
            w = ws[i] if i < len(ws) else np.nan
            data.append(w if not np.isnan(w) else 1.0)

    rows = np.array(rows, dtype=np.int64)
    cols = np.array(cols, dtype=np.int64)
    data = np.array(data, dtype=np.float64)

    # Build a presence set from original directed edges
    presence = set((int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist()))

    result: Dict[Tuple[int, int], float] = {}
    for (i, j) in presence:
        if i == j:
            # self-loops should be zero per user request
            w = 0.0
            result[(i, j)] = w
            # don't duplicate for (j,i) since it's the same
        elif (j, i) in presence:
            w = 2.0
            # mutual: set both directions to 2.0
            result[(i, j)] = w
            result[(j, i)] = w
        else:
            w = 1.0
            result[(i, j)] = w
            result[(j, i)] = w

    out_rows = []
    out_cols = []
    out_data = []
    for (r, c), v in result.items():
        out_rows.append(r)
        out_cols.append(c)
        out_data.append(v)

    rows = np.array(out_rows, dtype=np.int64)
    cols = np.array(out_cols, dtype=np.int64)
    data = np.array(out_data, dtype=np.float64)

    order = np.lexsort((cols, rows))
    rows = rows[order]
    cols = cols[order]
    data = data[order]

    # ensure self-loop entries have zero weight
    mask_self = rows == cols
    if mask_self.any():
        data[mask_self] = 0.0

    # remove zero-valued entries (drop self-loops entirely)
    keep = data != 0.0
    if not np.all(keep):
        rows = rows[keep]
        cols = cols[keep]
        data = data[keep]

    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=n), out=indptr[1:])
    return indptr, cols, data


def main():
    p = argparse.ArgumentParser(description="Build adjacency matrix (CSR) from neighbor TXT files.")
    p.add_argument("input", help="path to neighbor TXT file")
    p.add_argument("--symmetric", action="store_true", help="symmetrize (make undirected)")
    p.add_argument("--mutual", action="store_true", help="build weights as 1 if one-way, 2 if mutual (symmetrized)")
    p.add_argument("--distance", action="store_true", help="use numeric distances from file as weights instead of mutual binary weights")
    args = p.parse_args()

    inp = args.input
    if not os.path.exists(inp):
        raise SystemExit(f"File not found: {inp}")

    neighbors, weights = parse_neighbor_file(inp)
    print(f"Parsed {len(neighbors)} queries. Building adjacency...")

    # Default behavior: mutual weighting (1 if single-direction, 2 if mutual)
    weight_mode = "mutual" if not args.distance else "distance"
    adj = build_csr_from_neighbors(neighbors, weights, symmetric=args.symmetric,)


    # vertex weights all 1
    indptr, indices, data = adj
    vwgt = np.ones(indptr.shape[0]-1, dtype=np.int32)

    # Kahip expects integer arrays for adjacency pointers, adjacency indices and edge weights.
    # Cast to int32 to avoid pybind11 casting errors.
    xadj = indptr.astype(np.int32)
    adjncy = indices.astype(np.int32)

    adjwgt = data.astype(np.int32)

    N_BLOCKS = 10
    IMBALANCE = 0.03
    SEED = 42

    # Call kahip partitioner
    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjwgt, adjncy, N_BLOCKS, IMBALANCE, True, SEED, 1)
    print(f"Partitioned graph into {N_BLOCKS} blocks with edgecut {edgecut}.")
    print(blocks[:20])
    

    
    
    # Save the three CSR arrays (no SciPy dependency)


if __name__ == "__main__":
    main()
