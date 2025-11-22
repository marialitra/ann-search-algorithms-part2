#!/usr/bin/env python3
"""
Benchmark driver for running build + search experiments and saving results to CSV.

Usage: python3 scripts/benchmark_runs.py [--out results.csv] [--limit N]

Notes:
- This script runs `src/nlsh_build.py` to build and `src/nlsh_search.py` to search.
- It will, for each build configuration, run builds and then run searches for each T.
- When a build run would invoke Ivfflat (knngraph missing), the script runs the build twice
  (first with Ivfflat generating the knngraph, then again without) and records both times.
- When search needs to compute true neighbors (cache missing), the script runs the search twice
  (first to compute & cache true neighbors, then again to run with cache) and records both times.

Be careful: the full Cartesian product of parameters is large; use --limit to cap total iterations.
"""

import argparse
import csv
import itertools
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
DATA_MNIST = ROOT / 'Data' / 'SIFT' / 'sift_base.fvecs'
Q_MNIST = ROOT / 'Data' / 'SIFT' / 'sift_query.fvecs'
INDEX_DIR = ROOT / 'nlsh_index_sift'
OUTPUT_TXT = ROOT / 'output.txt'
KNNS_DIR = ROOT / 'knngraphs'
TRUE_NEIGH_DIR = ROOT / 'True Neighbors'

# Defaults requested by user
DEFAULTS = {
    'dataset': str(DATA_MNIST),
    'query': str(Q_MNIST),
    'index': str(INDEX_DIR),
    'type': 'SIFT',
    'R': 300.0,
}

# Parameter grids
KNN_LIST = [1, 5, 50, 100]
BLOCKS_LIST = [200, 2000, 5000, 10000]
IMBALANCE_LIST = [0.03, 0.1, 0.15]
KAHIP_MODE_LIST = [0, 1]
LAYERS_LIST = [2, 3, 5, 6]
NODES_LIST = [64, 128, 256]
EPOCHS_LIST = [1, 3, 5, 10]
BATCH_LIST = [1024, 2048, 4096]
T_LIST = [2, 50, 90, 100]

# Helper: slug for knngraph filename
def slug(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = s.strip('_')
    return s or 'unknown'

def knngraph_path(dataset: str, knn: int,blocks: int,imbalance: int,kahip_mode: int,layers: int,nodes: int,epochs: int,batch_size: int) -> Path:
    if imbalance == 0.1:
        name = f"knngraph_{slug(dataset)}_N{knn}_B{blocks}_I0_1_K_{kahip_mode}_L_{layers}_NOD_{nodes}_E_{epochs}_BS_{batch_size}.txt"
    if imbalance == 0.03:
        name = f"knngraph_{slug(dataset)}_N{knn}_B{blocks}_I0_03_K_{kahip_mode}_L_{layers}_NOD_{nodes}_E_{epochs}_BS_{batch_size}.txt"
    if imbalance == 0.15:
        name = f"knngraph_{slug(dataset)}_N{knn}_B{blocks}_I0_15_K_{kahip_mode}_L_{layers}_NOD_{nodes}_E_{epochs}_BS_{batch_size}.txt"
    return KNNS_DIR / name

# True neighbors cache paths (metadata path pattern)
def true_neighbors_meta_path(dataset: str, query: str, N: int) -> Path:
    fname = f"true_neighbors_{slug(dataset)}_{slug(query)}_N{int(N)}.meta.json"
    return TRUE_NEIGH_DIR / fname

# # Run command wrapper
# def run_cmd(cmd, env=None, cwd=ROOT, timeout=None):
#     start = time.perf_counter()
#     process = subprocess.Popen(
#         cmd,
#         cwd=cwd,
#         env=env,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         text=True,
#         bufsize=1  # line-buffered
#     )

#     stdout_log = []
#     stderr_log = []

#     try:
#         # Read stdout/stderr line-by-line as they come
#         for stream, collector in [
#             (process.stdout, stdout_log),
#         ]:
#             for line in iter(stream.readline, ''):
#                 print(line, end='')       # <-- LIVE output to your terminal
#                 collector.append(line)

#         # Wait for process to exit and capture remaining stderr
#         stderr_output = process.stderr.read()
#         if stderr_output:
#             print(stderr_output, end='')  # show errors live
#             stderr_log.append(stderr_output)

#         rc = process.wait(timeout=timeout)
#         end = time.perf_counter()

#         return rc, ''.join(stdout_log), ''.join(stderr_log), end - start

#     except subprocess.TimeoutExpired:
#         process.kill()
#         return 124, ''.join(stdout_log), ''.join(stderr_log), time.perf_counter() - start


def run_cmd(cmd, env=None, cwd=ROOT, timeout=None):
    start = time.perf_counter()
    try:
        r = subprocess.run(cmd, shell=False, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
        end = time.perf_counter()
        return r.returncode, r.stdout, r.stderr, end - start
    except subprocess.TimeoutExpired:
        return 124, '', 'timeout', time.perf_counter() - start



# Parse metrics from output.txt written by nlsh_search.py
def parse_search_output(output_file: Path):
    metrics = {'avg_AF': None, 'recall': None, 'QPS': None, 'tApprox': None, 'tTrue': None}
    if not output_file.exists():
        return metrics
    text = output_file.read_text()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('Average AF:'):
            metrics['avg_AF'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('Recall@'):
            metrics['recall'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('QPS:'):
            metrics['QPS'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('tApproximateAverage:'):
            metrics['tApprox'] = float(line.split(':', 1)[1].strip())
        elif line.startswith('tTrueAverage:'):
            metrics['tTrue'] = float(line.split(':', 1)[1].strip())
    return metrics

# Main driver

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT / 'benchmark_results_SIFT.csv'), help='CSV output path')
    p.add_argument('--limit', type=int, default=0, help='Optional limit on number of builds (0 = no limit)')
    p.add_argument('--dry', action='store_true', help='Dry run: just print what would be run')
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare CSV header
    header = [
        'knn','blocks','imbalance','kahip_mode','layers','nodes','epochs','batch_size','T',
        'build_time_with_ivfflat','build_time_without_ivfflat',
        'search_time_with_true','search_time_without_true','true_compute_time',
        'recall','avg_AF','QPS','tApprox'
    ]

    # Open CSV and append rows incrementally
    write_header = not out_path.exists()
    csvf = open(out_path, 'a', newline='')
    writer = csv.DictWriter(csvf, fieldnames=header)
    if write_header:
        writer.writeheader()
        csvf.flush()

    # Build a list of single-parameter experiments: vary one parameter while
    # keeping all others at baseline DEFAULTS (one-factor-at-a-time).
    param_grid = {
        'knn': KNN_LIST,
        'blocks': BLOCKS_LIST,
        'imbalance': IMBALANCE_LIST,
        'kahip_mode': KAHIP_MODE_LIST,
        'layers': LAYERS_LIST,
        'nodes': NODES_LIST,
        'epochs': EPOCHS_LIST,
        'batch_size': BATCH_LIST
    }

    env = os.environ.copy()

    build_idx = 0
    # Iterate param by param, varying only that param and reverting to baseline
    for pname, plist in param_grid.items():
        print(f"\n== Varying parameter: {pname} (baseline kept for others) ==")
        for val in plist:
            build_idx += 1
            # Baseline values come from the Makefile-like defaults
            baseline = {
                'knn': 1,
                'blocks': 2000,
                'imbalance': 0.1,
                'kahip_mode': 0,
                'layers': 3,
                'nodes': 128,
                'epochs': 3,
                'batch_size': 1024
            }

            # use baseline and override only pname
            cfg = baseline.copy()
            cfg[pname] = val

            knn = cfg['knn']
            blocks = cfg['blocks']
            imbalance = cfg['imbalance']
            kahip_mode = cfg['kahip_mode']
            layers = cfg['layers']
            nodes = cfg['nodes']
            epochs = cfg['epochs']
            batch_size = cfg['batch_size']

            print(f"\n=== Build {build_idx}: {pname}={val} (knn={knn} blocks={blocks} layers={layers} nodes={nodes} epochs={epochs} batch={batch_size}) ===")

            build_cmd = [sys.executable, str(SRC / 'nlsh_build.py'),
                         '-d', DEFAULTS['dataset'],
                         '-i', str(INDEX_DIR),
                         '--type', DEFAULTS['type'],
                         '--knn', str(knn),
                         '-m', str(blocks),
                         '--imbalance', str(imbalance),
                         '--kahip_mode', str(kahip_mode),
                         '--layers', str(layers),
                         '--nodes', str(nodes),
                         '--epochs', str(epochs),
                         '--batch_size', str(batch_size),
                         '--lr', str(DEFAULTS.get('lr', 0.001)),
                         '--seed', str(DEFAULTS.get('seed', 42))]

            knnpath = knngraph_path(DEFAULTS['dataset'], knn,blocks,imbalance,kahip_mode,layers,nodes,epochs,batch_size)
            KNNS_DIR.mkdir(parents=True, exist_ok=True)

            build_time_with = None
            build_time_without = None

            if args.dry:
                print('DRY RUN build cmd:', ' '.join(build_cmd))
                # Force Ivfflat run by removing knngraph (ONLY WHEN USING THE DEFAULTS THIS WILL HAPPEN)
                if knnpath.exists():
                    try:
                        # knnpath.unlink() 
                        print('HEREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
                        continue   
                    except Exception:
                        pass
                else:
                    print('HEEEEEEEEEEEEELLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL')
                
            else:
                # Force Ivfflat run by removing knngraph (ONLY WHEN USING THE DEFAULTS THIS WILL HAPPEN)
                if knnpath.exists():
                    try:
                        # knnpath.unlink() 
                        continue   
                    except Exception:
                        pass

                print('Running build (with Ivfflat generation) ...')
                rc, out, err, bt = run_cmd(build_cmd, env=env)
                build_time_with = bt
                (ROOT / f'SIFT_build_out_{build_idx}_with.log').write_text(out + '\n' + err)

                print('Running build again (knngraph present) ...')
                rc2, out2, err2, bt2 = run_cmd(build_cmd, env=env)
                build_time_without = bt2
                (ROOT / f'SIFT_build_out_{build_idx}_without.log').write_text(out2 + '\n' + err2)

            # Run searches for all T values for this build
            for T in T_LIST:
                print(f"--- Search for T={T} ---")
                search_cmd = [sys.executable, str(SRC / 'nlsh_search.py'),
                              '-d', DEFAULTS['dataset'],
                              '-q', DEFAULTS['query'],
                              '-i', str(INDEX_DIR),
                              '-o', str(OUTPUT_TXT),
                              '-type', DEFAULTS['type'],
                              '-N', str(knn),
                              '-R', str(DEFAULTS['R']),
                              '-T', str(T),
                              '-range', 'false']

                if args.dry:
                    print('DRY RUN search cmd:', ' '.join(search_cmd))
                    row = {k: None for k in header}
                    row.update({'knn': knn, 'blocks': blocks, 'imbalance': imbalance, 'kahip_mode': kahip_mode,
                                'layers': layers, 'nodes': nodes, 'epochs': epochs, 'batch_size': batch_size, 'T': T})
                    writer.writerow(row)
                    csvf.flush()
                    continue

                # Remove true neighbors cache to force brute-force compute first run
                meta_path = true_neighbors_meta_path(DEFAULTS['dataset'], DEFAULTS['query'], knn)
                if meta_path.exists():
                    try:
                        arr_path = meta_path.with_suffix('.npy')
                        meta_path.unlink()
                        if arr_path.exists():
                            arr_path.unlink()
                    except Exception:
                        pass

                rc_s1, out_s1, err_s1, st1 = run_cmd([env.get('PYTHON', sys.executable)] + search_cmd[1:], env=env)
                search_time_with = st1
                (ROOT / f'SIFT_search_out_build{build_idx}_T{T}_with.log').write_text(out_s1 + '\n' + err_s1)

                true_compute_time = None
                meta_path = true_neighbors_meta_path(DEFAULTS['dataset'], DEFAULTS['query'], knn)
                if meta_path.exists():
                    try:
                        import json
                        meta = json.load(open(meta_path))
                        true_compute_time = float(meta.get('time_seconds', 0.0))
                    except Exception:
                        true_compute_time = None

                metrics1 = parse_search_output(OUTPUT_TXT)

                rc_s2, out_s2, err_s2, st2 = run_cmd([env.get('PYTHON', sys.executable)] + search_cmd[1:], env=env)
                search_time_without = st2
                (ROOT / f'SIFT_search_out_build{build_idx}_T{T}_without.log').write_text(out_s2 + '\n' + err_s2)
                metrics2 = parse_search_output(OUTPUT_TXT)

                metrics_used = metrics2 if metrics2.get('recall') is not None else metrics1

                row = {
                    'knn': knn,
                    'blocks': blocks,
                    'imbalance': imbalance,
                    'kahip_mode': kahip_mode,
                    'layers': layers,
                    'nodes': nodes,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'T': T,
                    'build_time_with_ivfflat': build_time_with,
                    'build_time_without_ivfflat': build_time_without,
                    'search_time_with_true': search_time_with,
                    'search_time_without_true': search_time_without,
                    'true_compute_time': true_compute_time,
                    'recall': metrics_used.get('recall'),
                    'avg_AF': metrics_used.get('avg_AF'),
                    'QPS': metrics_used.get('QPS'),
                    'tApprox': metrics_used.get('tApprox')
                }

                writer.writerow(row)
                csvf.flush()

    csvf.close()
    print(f"Done. Results appended to {out_path}")

if __name__ == '__main__':
    main()
