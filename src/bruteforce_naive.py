import libraries
from libraries import np, time

# Minimal, intentionally unoptimized brute-force implementation.
# - Uses pure Python loops to compute Euclidean distances (no vectorized NumPy ops)
# - Sorts distance lists with Python's `sorted` and picks top-N
# - Includes a simple caching mechanism similar to the optimized bruteforce module

def _make_cache_paths(base_dir: str, datasetName: str, queryName: str, N: int):
    base_dir = base_dir or "."
    target_dir = libraries.os.path.join(base_dir, "True Neighbors")
    libraries.os.makedirs(target_dir, exist_ok=True)
    fname = f"true_neighbors_naive_{datasetName}_{queryName}_N{int(N)}.npy"
    meta = f"true_neighbors_naive_{datasetName}_{queryName}_N{int(N)}.meta.json"
    return libraries.os.path.join(target_dir, fname), libraries.os.path.join(target_dir, meta)


def find_and_save_true_neighbors_naive(X_flat: np.ndarray, Q_flat: np.ndarray, N: int,
                                      true_neighbors_file: str, datasetName: str, queryName: str):
    """
    Naive O(N * Q * D) brute-force top-N neighbors computed in plain Python loops.
    Returns (true_neighbors_indices, elapsed_time)
    """
    if true_neighbors_file is None or libraries.os.path.isdir(true_neighbors_file):
        arr_path, meta_path = _make_cache_paths(true_neighbors_file, datasetName, queryName, N)
    else:
        arr_path = true_neighbors_file
        meta_path = libraries.os.path.splitext(arr_path)[0] + ".meta.json"

    print("[naive-bruteforce] Computing true neighbors (very slow, unoptimized)...")

    # Flatten & ensure Python lists for naive loops
    X = X_flat.reshape(X_flat.shape[0], -1).astype(float)
    Q = Q_flat.reshape(Q_flat.shape[0], -1).astype(float)

    nq, dimq = Q.shape
    n, dimx = X.shape
    assert dimq == dimx

    true_idx = np.empty((nq, N), dtype=np.int64)

    t0 = time.perf_counter()

    # Naive triple-loop: for each query, compute distance to every X using inner-loop over dimensions
    for qi in range(nq):
        qv = Q[qi]
        dists = []  # list of (dist, idx)
        for xi in range(n):
            xv = X[xi]
            s = 0.0
            # inner loop over dimensions (explicit Python loop)
            for k in range(dimq):
                diff = qv[k] - xv[k]
                s += diff * diff
            dists.append((s, xi))

        # sort by distance (Python Timsort) and take top-N
        dists.sort(key=lambda t: t[0])
        topN = [idx for (_, idx) in dists[:N]]
        # store
        for j in range(N):
            true_idx[qi, j] = int(topN[j]) if j < len(topN) else -1

    t1 = time.perf_counter()
    elapsed = t1 - t0

    # Save results and metadata
    try:
        np.save(arr_path, true_idx)
        meta = {
            "dataset_name": str(datasetName),
            "query_name": str(queryName),
            "N": int(N),
            "dataset_size": int(n),
            "query_size": int(nq),
            "dim": int(dimq),
            "time_seconds": float(elapsed),
            "note": "naive unoptimized implementation"
        }
        with open(meta_path, "w") as fh:
            libraries.json.dump(meta, fh, indent=2)
        print(f"[naive-bruteforce] Saved naive true neighbors to {arr_path}")
    except Exception as e:
        print(f"[naive-bruteforce] Warning: could not save cache: {e}")

    return true_idx, elapsed


def load_or_compute_true_neighbors_naive(X: np.ndarray, Q: np.ndarray, datasetName: str, queryName: str,
                                         N: int, true_neighbors_file: str = None, cache_dir: str = "."):
    X_flat = X.reshape(X.shape[0], -1)
    Q_flat = Q.reshape(Q.shape[0], -1)

    if true_neighbors_file is None or libraries.os.path.isdir(true_neighbors_file):
        arr_path, meta_path = _make_cache_paths(cache_dir, datasetName, queryName, N)
    else:
        arr_path = true_neighbors_file
        meta_path = libraries.os.path.splitext(arr_path)[0] + ".meta.json"

    if libraries.os.path.exists(arr_path) and libraries.os.path.exists(meta_path):
        try:
            meta = libraries.json.load(open(meta_path))
            t_saved = float(meta.get("time_seconds", 0.0))
            arr = np.load(arr_path)
            print(f"[naive-bruteforce] Loaded cached naive true neighbors from {arr_path} (t={t_saved:.6f}s)")
            return arr, t_saved
        except Exception:
            print("[naive-bruteforce] Cache invalid or unreadable, recomputing...")
            return find_and_save_true_neighbors_naive(X_flat, Q_flat, N, arr_path, datasetName, queryName)
    else:
        return find_and_save_true_neighbors_naive(X_flat, Q_flat, N, arr_path, datasetName, queryName)


def calculate_recall_naive(true_neighbors_array: np.ndarray, all_lsh_neighbors: list, N: int) -> float:
    # simple recall as in optimized module
    total_queries = true_neighbors_array.shape[0]
    total_recall = 0.0
    for qi in range(total_queries):
        true_set = set(true_neighbors_array[qi])
        lsh_set = set(all_lsh_neighbors[qi])
        hits = len(true_set.intersection(lsh_set))
        total_recall += hits / N
    avg = total_recall / total_queries
    print(f"[naive-bruteforce] Average Recall@{N}: {avg:.4f}")
    return avg
