#!/usr/bin/env python3
"""Visualize a graph described by CSR lists in a text file.

The text file should contain four named lists: xadj, adjncy, adjwgt, vwgt
formatted like the `a.txt` attachment in the workspace, e.g.:

xadj:
[ 0  6 10 ... ]
adjncy
[10 7 8 ...]
adjwgt
[2 2 1 ...]
vwgt
[1 1 1 ...]

This script parses those lists and draws the graph to a PNG using networkx.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List

import numpy as np


def parse_bracket_array(lines: List[str], start_idx: int) -> (np.ndarray, int):
    """Read lines starting at start_idx and parse a bracketed array which may span lines.
    Returns (array, next_index_after_array).
    """
    buf = ""
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        buf += ' ' + line
        if ']' in line:
            break
        i += 1

    # extract the bracketed part
    m = re.search(r"\[(.*)\]", buf, flags=re.S)
    if not m:
        raise ValueError("Could not parse bracketed array at lines starting %d" % start_idx)
    body = m.group(1)
    # replace commas with spaces and collapse whitespace
    body = body.replace(',', ' ')
    body = re.sub(r"\s+", ' ', body).strip()
    if body == '':
        arr = np.array([], dtype=np.int64)
    else:
        # Use fromstring for speed
        arr = np.fromstring(body, sep=' ', dtype=np.int64)
    return arr, i + 1


def parse_csr_file(path: str):
    with open(path, 'r', encoding='utf-8') as fh:
        lines = fh.readlines()

    i = 0
    xadj = adjncy = adjwgt = vwgt = None
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('xadj'):
            # next lines contain bracketed array
            # skip the label line if it contains ':'
            if ':' in line and '[' in line:
                # bracket on same line
                arr, i = parse_bracket_array(lines, i)
            else:
                # bracket on next lines
                arr, i = parse_bracket_array(lines, i + 1)
            xadj = arr
            continue
        if line.startswith('adjncy'):
            # same pattern
            if '[' in line:
                arr, i = parse_bracket_array(lines, i)
            else:
                arr, i = parse_bracket_array(lines, i + 1)
            adjncy = arr
            continue
        if line.startswith('adjwgt'):
            if '[' in line:
                arr, i = parse_bracket_array(lines, i)
            else:
                arr, i = parse_bracket_array(lines, i + 1)
            adjwgt = arr
            continue
        if line.startswith('vwgt'):
            if '[' in line:
                arr, i = parse_bracket_array(lines, i)
            else:
                arr, i = parse_bracket_array(lines, i + 1)
            vwgt = arr
            continue
        i += 1

    if xadj is None or adjncy is None or adjwgt is None or vwgt is None:
        raise ValueError('Missing one of xadj/adjncy/adjwgt/vwgt in file')

    return xadj.astype(np.int64), adjncy.astype(np.int64), adjwgt.astype(np.int64), vwgt.astype(np.int64)


def draw_csr(xadj, adjncy, adjwgt, vwgt, outfile='csr_graph.png'):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception:
        raise SystemExit('Requires networkx and matplotlib: pip install networkx matplotlib')

    n = len(vwgt)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    # Build edges
    for u in range(n):
        start = xadj[u]
        end = xadj[u+1]
        for idx in range(start, end):
            v = int(adjncy[idx])
            w = int(adjwgt[idx])
            if u == v:
                continue
            if G.has_edge(u, v):
                G[u][v]['weight'] += w
            else:
                G.add_edge(u, v, weight=w)

    # visual parameters
    node_sizes = [50 + 100 * int(w) for w in vwgt]
    weights = [d.get('weight', 1) for u, v, d in G.edges(data=True)]
    maxw = max(weights) if weights else 1
    widths = [max(0.2, 2.0 * (float(w) / maxw)) for w in weights]

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='C0')
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser(description='Visualize CSR lists stored in a text file')
    p.add_argument('input', help='path to text file containing xadj/adjncy/adjwgt/vwgt')
    p.add_argument('--outfile', default='csr_graph.png')
    args = p.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit('File not found: %s' % args.input)

    xadj, adjncy, adjwgt, vwgt = parse_csr_file(args.input)
    print('Parsed CSR: n=', len(vwgt), 'nnz=', len(adjncy))
    draw_csr(xadj, adjncy, adjwgt, vwgt, outfile=args.outfile)
    print('Saved', args.outfile)


if __name__ == '__main__':
    main()
