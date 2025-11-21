import libraries
from libraries import np, List, Tuple, _slug, time

# --- Brute-Force and Caching Logic ---

def _make_cache_paths(base_dir: str, datasetName: str, queryName: str, N: int) -> Tuple[str, str] :
    """
        Returns (array_path, meta_path) for the cache files given dataset/query names and N.
        base_dir can be a folder; if empty, uses current folder.
        The files are placed in a subfolder "True Neighbors" with formatted names as:
        true_neighbors_{datasetName}_{queryName}_N{N}.npy
        true_neighbors_{datasetName}_{queryName}_N{N}.meta.json
    """

    base_dir = base_dir or "."
    target_dir = libraries.os.path.join(base_dir, "True Neighbors")
    libraries.os.makedirs(target_dir, exist_ok=True)

    libraries.os.makedirs(base_dir, exist_ok=True)
    fname = f"true_neighbors_{_slug(datasetName)}_{_slug(queryName)}_N{int(N)}.npy"
    meta = f"true_neighbors_{_slug(datasetName)}_{_slug(queryName)}_N{int(N)}.meta.json"
    return libraries.os.path.join(target_dir, fname), libraries.os.path.join(target_dir, meta)

def find_and_save_true_neighbors(X_flat: np.ndarray, Q_flat: np.ndarray, N: int,
                                 true_neighbors_file: str, datasetName: str, queryName: str) -> np.ndarray:
    """
        Compute brute-force top-N neighbors (exact) and save array and metadata.
        X_flat: (n, d), Q_flat: (nq, d)
        true_neighbors_file: path to .npy file (if None, will be auto-generated)
        datasetName, queryName: used for metadata & auto filename
    """
    # If user gave a directory or None, auto-generate filename in that directory
    if true_neighbors_file is None or libraries.os.path.isdir(true_neighbors_file):
        arr_path, meta_path = _make_cache_paths(true_neighbors_file, datasetName, queryName, N)
    else:
        # Use given path; meta file placed next to it
        arr_path = true_neighbors_file
        meta_path = libraries.os.path.splitext(arr_path)[0] + ".meta.json"

    print("-" * 50)
    print(f"Brute-Force: Finding True Neighbors (N={N}) and Caching to '{arr_path}'...")

    # Memory-efficient block-wise computation of top-N neighbors.
    # Process queries in small batches and dataset in chunks so we never allocate
    # The full (nq x n) distance matrix which can be huge.
    nq, dimq = Q_flat.shape
    n, dimx = X_flat.shape
    assert dimq == dimx, "Query and dataset dimensions must match"

    # Cast to float32 to save memory (if not already)
    X_arr = X_flat.astype(np.float32, copy=False)
    Q_arr = Q_flat.astype(np.float32, copy=False)

    # Process in chunks to save memory
    X_CHUNK = 10000  # Number of dataset vectors per chunk
    Q_BATCH = 128    # Number of queries per batch

    # Output arrays: indices and squared distances
    top_idx = np.empty((nq, N), dtype=np.int64)
    top_dist = np.full((nq, N), np.inf, dtype=np.float32)

    start_time = time.perf_counter()

    # Compute squared norms of dataset in chunks on the fly
    for qi in range(0, nq, Q_BATCH):
        q_slice = slice(qi, min(qi + Q_BATCH, nq))
        Qb = Q_arr[q_slice]
        Qb_norm_sq = np.sum(Qb * Qb, axis=1)[:, None]

        # For each dataset chunk, compute distances to this Q batch and update top-N
        for xi in range(0, n, X_CHUNK):
            x_slice = slice(xi, min(xi + X_CHUNK, n))
            Xc = X_arr[x_slice]
            Xc_norm_sq = np.sum(Xc * Xc, axis=1)[None, :]

            # Compute squared Euclidean distances: ||q - x||^2 = ||q||^2 - 2q.x + ||x||^2

            prod = Qb.astype(np.float32) @ Xc.T
            dist_sq = Qb_norm_sq - 2.0 * prod + Xc_norm_sq

            # For each query in batch, find local top-N indices within this chunk
            # Then merge with existing global top-N.
            # Use argpartition for efficiency
            if dist_sq.size == 0:
                continue
            kth = min(N, dist_sq.shape[1] - 1)
            local_idx = np.argpartition(dist_sq, kth, axis=1)[:, :N] 

            # Gather distances and convert local indices to global indices
            rows = np.arange(dist_sq.shape[0])[:, None]
            local_dists = dist_sq[rows, local_idx]
            global_indices = (local_idx + xi).astype(np.int64)

            # Merge local top-N with global top-N for this batch range
            for j in range(dist_sq.shape[0]):
                qj = qi + j

                # Candidates: existing top + new local
                cand_dists = np.concatenate([top_dist[qj], local_dists[j]])
                cand_idx = np.concatenate([top_idx[qj], global_indices[j]])

                if cand_dists.size <= N:
                    order = np.argsort(cand_dists)
                else:
                    order = np.argpartition(cand_dists, N)[:N]
                    order = order[np.argsort(cand_dists[order])]

                top_dist[qj] = cand_dists[order]
                top_idx[qj] = cand_idx[order]

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # After processing all chunks, top_idx contains indices of nearest neighbors
    true_neighbors_indices = top_idx.astype(np.int64)

    # Save array and meta
    try:
        np.save(arr_path, true_neighbors_indices)
        meta = {
            "dataset_name": str(datasetName),
            "query_name": str(queryName),
            "N": int(N),
            "dataset_size": int(n),
            "query_size": int(nq),
            "dim": int(dimq),
            "time_seconds": elapsed_time  # Stores total computation time
        }
        with open(meta_path, "w") as fh:
            libraries.json.dump(meta, fh, indent=2)
        print(f"Saved true neighbors to {arr_path}")
        print(f"Saved meta to {meta_path}")
    except Exception as e:
        print(f"Warning: could not save true neighbors or meta: {e}")

    print("-" * 50)
    return true_neighbors_indices, elapsed_time

def load_or_compute_true_neighbors(X: np.ndarray, Q: np.ndarray, datasetName: str, queryName: str,
                                   N: int, true_neighbors_file: str = None, cache_dir: str = ".") -> np.ndarray:
    """
        Loads cached true neighbors if available, else computes and saves anew.
        - If true_neighbors_file is provided and is a filepath it will be used.
        - If true_neighbors_file is None or points to a directory, the cache filename is auto-generated in cache_dir.
    """
    # Flatten inputs
    X_flat = X.reshape(X.shape[0], -1)
    Q_flat = Q.reshape(Q.shape[0], -1)

    # Determine paths
    if true_neighbors_file is None or libraries.os.path.isdir(true_neighbors_file):
        arr_path, meta_path = _make_cache_paths(cache_dir, datasetName, queryName, N)
    else:
        arr_path = true_neighbors_file
        meta_path = libraries.os.path.splitext(arr_path)[0] + ".meta.json"

    # If cache exists, validate meta
    if libraries.os.path.exists(arr_path) and libraries.os.path.exists(meta_path):
        try:
            meta = libraries.json.load(open(meta_path))

            # Check N, sizes and names to see if we have to recompute the neighbors!
            cached_N = int(meta.get("N", -1))
            cached_qsize = int(meta.get("query_size", -1))
            cached_dsize = int(meta.get("dataset_size", -1))
            cached_dname = str(meta.get("dataset_name", ""))
            cached_qname = str(meta.get("query_name", ""))

            incompatible_reasons = []
            if cached_N != N:
                incompatible_reasons.append(f"N mismatch (cache {cached_N} != requested {N})")
            if cached_qsize != Q_flat.shape[0]:
                incompatible_reasons.append(f"query count mismatch (cache {cached_qsize} != {Q_flat.shape[0]})")
            if cached_dsize != X_flat.shape[0]:
                incompatible_reasons.append(f"dataset count mismatch (cache {cached_dsize} != {X_flat.shape[0]})")
            if datasetName and cached_dname != str(datasetName):
                incompatible_reasons.append(f"dataset name mismatch (cache '{cached_dname}' != '{datasetName}')")
            if queryName and cached_qname != str(queryName):
                incompatible_reasons.append(f"query name mismatch (cache '{cached_qname}' != '{queryName}')")

            if incompatible_reasons:
                print("Cached true-neighbors file found but incompatible for these reasons:")
                for r in incompatible_reasons:
                    print("  -", r)
                print("Recomputing true neighbors...")
                return find_and_save_true_neighbors(X_flat, Q_flat, N, arr_path, datasetName, queryName)

            # Load array
            true_neighbors = np.load(arr_path)

            # Sanity shape check
            if true_neighbors.ndim != 2 or true_neighbors.shape[0] != Q_flat.shape[0] or true_neighbors.shape[1] != N:
                print("Cached array shape mismatch; recomputing...")
                return find_and_save_true_neighbors(X_flat, Q_flat, N, arr_path, datasetName, queryName)

            t_true = float(meta.get("time_seconds", 0.0))

            # Debugging Reasons
            # print(f"Loaded cached true neighbors from {arr_path} (original compute time t={t_true:.6f}s)")

            return true_neighbors, t_true

        except Exception as e:
            print(f"Error reading cache files ({e}); recomputing...")
            return find_and_save_true_neighbors(X_flat, Q_flat, N, arr_path, datasetName, queryName)
    else:
        # Cache not present
        return find_and_save_true_neighbors(X_flat, Q_flat, N, arr_path, datasetName, queryName)

def calculate_recall(true_neighbors_array: np.ndarray, all_lsh_neighbors: List[np.ndarray], N: int) -> float:
    """
        Calculates Recall@N given the true neighbors and the LSH results.
    """

    print("-" * 50)
    print(f"Starting Recall@{N} Evaluation...")
    
    total_queries = true_neighbors_array.shape[0]
    total_recall = 0.0

    if total_queries != len(all_lsh_neighbors):
        raise ValueError("Mismatch between true neighbor array size and LSH result size.")

    for qi in range(total_queries):
        true_topN_indices = set(true_neighbors_array[qi]) 
        
        # LSH neighbors are the indices retrieved by the search
        lsh_neighbors = set(all_lsh_neighbors[qi])
        
        # Intersection: True Positives found by LSH 
        hits = len(true_topN_indices.intersection(lsh_neighbors))
        
        # Recall@N: (True Positives) / N
        recall_q = hits / N
        total_recall += recall_q
    
    average_recall = total_recall / total_queries
    
    print(f"Evaluation Completed. Average Recall@{N}: {average_recall:.4f}")
    print("-" * 50)
    
    return average_recall