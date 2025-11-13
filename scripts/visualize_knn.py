#!/usr/bin/env python3
"""Visualize a KNN neighbor graph stored in the text format used in this repo.

Example format:
Query: 0
Nearest neighbor-1: 0
distanceApproximate: 0.000000
Nearest neighbor-2: 4800
distanceApproximate: 826.075057
...

This script builds a small graph (optionally sampling the first N queries) and
saves a PNG image using networkx + matplotlib. For large graphs you must use
--sample to limit the number of nodes drawn.

Usage:
  python3 scripts/visualize_knn.py path/to/file.txt --sample 300 --outfile graph.png --mutual
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np


def parse_neighbor_file(path: str) -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    q_re = re.compile(r"^Query:\s*(\d+)")
    nn_re = re.compile(r"^Nearest neighbor-\d+:\s*(\d+)")
    d_re = re.compile(r"^distanceApproximate:\s*([0-9.+-eE]+)")

    neighbors = {}
    weights = {}
    current_q = None
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
                weights[current_q].append(np.nan)
                continue
            m = d_re.match(line)
            if m and current_q is not None:
                val = float(m.group(1))
                if neighbors[current_q]:
                    weights[current_q][-1] = val
                continue
    return neighbors, weights


def build_small_graph(neighbors: Dict[int, List[int]], weights: Dict[int, List[float]],
                      max_nodes: int = 300, mutual: bool = True, distance: bool = False):
    """Return nodes list and edges list (u,v,weight) for nodes [0..max_nodes-1]."""
    nodes = sorted([n for n in neighbors.keys() if n < max_nodes])
    presence = set()
    for n in nodes:
        for nb in neighbors.get(n, []):
            if nb < max_nodes:
                presence.add((n, nb))

    edges = []
    if mutual:
        # mutual weighting: 2 if both directions present, 1 otherwise; self-loops removed
        for (i, j) in sorted(presence):
            if i == j:
                continue
            if (j, i) in presence:
                w = 2.0
            else:
                w = 1.0
            edges.append((i, j, w))
    elif distance:
        # use distance values where available
        for i in nodes:
            ws = weights.get(i, [])
            nbrs = neighbors.get(i, [])
            for k, j in enumerate(nbrs):
                if j >= max_nodes or i == j:
                    continue
                val = ws[k] if k < len(ws) and not np.isnan(ws[k]) else 1.0
                edges.append((i, j, float(val)))
    else:
        # binary directed edges
        for (i, j) in sorted(presence):
            if i == j:
                continue
            edges.append((i, j, 1.0))

    return nodes, edges


def draw_graph(nodes, edges, outfile: str, figsize=(10, 10), max_nodes_for_labels=200):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception:
        raise SystemExit("This script requires networkx and matplotlib. Install via: pip install networkx matplotlib")

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for u, v, w in edges:
        # for visualization, collapse parallel edges by summing weights
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    # choose layout
    n = len(nodes)
    plt.figure(figsize=figsize)
    if n <= 1000:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.random_layout(G)

    weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    # scale edge width
    maxw = max(weights) if weights else 1.0
    widths = [max(0.2, 2.0 * (w / maxw)) for w in weights]

    node_size = 50 if n > 500 else 200
    nx.draw_networkx_edges(G, pos, alpha=0.7, width=widths)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="C0")
    if n <= max_nodes_for_labels:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="neighbor TXT file")
    p.add_argument("--sample", type=int, default=300, help="number of queries (nodes) to include")
    p.add_argument("--outfile", default="knn_graph.png", help="output PNG file")
    p.add_argument("--mutual", action="store_true", help="use mutual weighting (2 if mutual, 1 otherwise)")
    p.add_argument("--distance", action="store_true", help="use numeric distances as weights")
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"File not found: {args.input}")

    neighbors, weights = parse_neighbor_file(args.input)
    nodes, edges = build_small_graph(neighbors, weights, max_nodes=args.sample, mutual=args.mutual, distance=args.distance)

    print(f"Building visualization for {len(nodes)} nodes and {len(edges)} edges -> {args.outfile}")
    draw_graph(nodes, edges, args.outfile)
    print("Saved", args.outfile)


if __name__ == "__main__":
    main()
