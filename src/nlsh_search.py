import libraries
from libraries import np, CNNClassifier, MLPClassifier, Tuple, List, load_idx_images, load_sift_vectors

from bruteforce import load_or_compute_true_neighbors, calculate_recall

Classifier = None

def main():
    p = libraries.argparse.ArgumentParser(description="Neural LSH search phase.")
    p.add_argument("-d", "--dataset", required=True, type=str, help="Dataset file")
    p.add_argument("-q", "--query", required=True, type=str, help="Query file")
    p.add_argument("-i", "--index", required=True, type=str, help="Path to built index directory")
    p.add_argument("-o", "--output", required=True, type=str, help="Output results file")
    p.add_argument("-type", required=True, type=str, help="Type of the given dataset (MNIST or SIFT1M)")
    p.add_argument("-N", type=int, default=1, help="Number of nearest neighbors to report")
    p.add_argument("-R", type=float, default=-2, help="Distance for range search if enable")
    p.add_argument("-T", type=int, default=5, help="Number of bins to probe (multi-probe)")
    p.add_argument("-range", type=str, default="true", help="Range search flag")
    args = p.parse_args()
    print(libraries.time.strftime("%Y-%m-%d %H:%M:%S", libraries.time.localtime()), "Starting Neural LSH search...")

    is_mnist = args.type and args.type.lower().startswith("mnist")
    is_sift = args.type and args.type.lower().startswith("sift")

    # --- Setup default R ---
    if args.R < 0:
        # Meaning that no given value has been given, or a wrong one detected,
        # We are going to use the defaut values for each dataset type

        if is_sift:
            args.R = 2800
        elif is_mnist:
            args.R = 2000
        else:
            print("Not acceptable dataset type")
            exit()

    # --- Setup ---
    print("Loading index ...")
    meta_path = libraries.os.path.join(args.index, "meta.json")
    
    # Read meta data, including new image dimensions
    if libraries.os.path.exists(meta_path):
        meta = libraries.json.load(open(meta_path))
        num_vectors = meta["n_vectors"]
        d_in = meta["dim"]
        m = meta["n_bins"]
        img_rows = meta["img_rows"]
        img_cols = meta["img_cols"]
    else:
        print("Warning: meta.json not found.")
        exit()

    model_path = libraries.os.path.join(args.index, "model.pth")
    inverted_path = libraries.os.path.join(args.index, "inverted_file.npy")
    
    model = None
    if libraries.os.path.exists(model_path):
        # Load the checkpoint first and decide which classifier to instantiate
        sd = libraries.torch.load(model_path, map_location="cpu")
        # Handle common checkpoint wrappers where the state_dict is nested
        if isinstance(sd, dict):
            # common keys used by training scripts
            for k in ("state_dict", "model_state_dict", "model", "net"): 
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break

        # Heuristic: if the saved state_dict contains 'conv1.0.weight' it's a CNN;
        # if it contains keys starting with 'net.' it's the MLP implementation.
        sd_keys = list(sd.keys()) if isinstance(sd, dict) else []

        if any(k.startswith("conv1") or k.startswith("conv1.0") for k in sd_keys):
            # CNN checkpoint
            model = CNNClassifier(img_rows=img_rows, img_cols=img_cols, n_out=m)
            model.load_state_dict(sd)
            model.eval()
        elif any(k.startswith("net.") for k in sd_keys):
            # MLP checkpoint: infer hidden size and number of linear layers
            # find keys of the form 'net.<i>.weight'
            linear_keys = [k for k in sd_keys if k.startswith("net.") and k.endswith(".weight")]
            if not linear_keys:
                raise RuntimeError("Saved state_dict looks like an MLP but no linear weights found.")
            # pick the first linear weight to infer hidden size and input dim
            first_w = sd[sorted(linear_keys)[0]]
            hidden_size = int(first_w.shape[0])
            # number of linear layers == number of linear_keys
            n_linears = len(linear_keys)
            # conlibraries.struct an MLP with that many linear layers
            model = MLPClassifier(d_in=d_in, n_out=m, hidden_size=hidden_size, n_layers=n_linears, dropout=0.0)
            model.load_state_dict(sd)
            model.eval()
        else:
            # fallback: try to initialize CNN but load with strict=False to allow partial matches
            try:
                model = CNNClassifier(img_rows=img_rows, img_cols=img_cols, n_out=m)
                model.load_state_dict(sd, strict=False)
                model.eval()
                print("Warning: loaded model.pth with fallback CNN (strict=False).")
            except Exception:
                raise RuntimeError("Could not infer model architecture from saved state_dict. Please ensure the saved model matches the search-time model.")
    else:
        print("Warning: model.pth not found. Skipping model loading/hashing.")

    inverted = {}
    if libraries.os.path.exists(inverted_path):
        inverted = np.load(inverted_path, allow_pickle=True).item()
        print(f"Loaded inverted file with {len(inverted)} bins.")
    else:
        print("Warning: inverted_file.npy not found. Skipping LSH search.")

    # --- Load Data and Normalize ---
    # Load dataset / queries depending on type. If the dataset is SIFT use the
    # SIFT loader (fvecs). Otherwise assume IDX images.
    if is_sift:
        print("Loading SIFT dataset and queries (fvecs) ...")

        X, num_images, num_rows, num_cols = load_sift_vectors(args.dataset)

        # Checks that everything is fine between the meta data and the actual data
        if img_rows != num_rows:
            raise ValueError(f"Invalid rows number in the images: {img_rows} != {num_rows}")
        
        if img_cols != num_cols:
            raise ValueError(f"Invalid columns number in the images: {img_cols} != {num_cols}")
            
        Q, num_images, num_rows, num_cols = load_sift_vectors(args.query)

        # Checks that everything is fine between the meta data and the actual data
        if img_rows != num_rows:
            raise ValueError(f"Invalid rows number in the query images: {img_rows} != {num_rows}")
        
        if img_cols != num_cols:
            raise ValueError(f"Invalid columns number in the query images: {img_cols} != {num_cols}")
            
        print(f"Loaded SIFT data shapes: X={X.shape}, Q={Q.shape}")
    else:
        print(f"Loading dataset and queries with shape (N, 1, {img_rows}, {img_cols}) ...")
        # load IDX images (MNIST)
        X, num_images, num_rows, num_cols = load_idx_images(args.dataset)

        # Checks that everything is fine between the meta data and the actual data
        if img_rows != num_rows:
            raise ValueError(f"Invalid rows number in the images: {img_rows} != {num_rows}")
        
        if img_cols != num_cols:
            raise ValueError(f"Invalid columns number in the images: {img_cols} != {num_cols}")
            
        Q, num_images, num_rows, num_cols = load_idx_images(args.query)

        # Checks that everything is fine between the query data and the actual MNIST data
        if img_rows != num_rows:
            raise ValueError(f"Invalid rows number in the query images: {img_rows} != {num_rows}")
        
        if img_cols != num_cols:
            raise ValueError(f"Invalid columns number in the query images: {img_cols} != {num_cols}")
            
    # Normalize depending on dataset type.
    # - For image/IDX data (MNIST) scale 0-255 -> 0.0-1.0
    # - For SIFT (fvecs) do L2 normalization per-vector to match training preprocessing
    if args.type and args.type.lower().startswith('sift'):
        # Ensure float32 and L2-normalize each vector (flattened)
        X = X.astype(np.float32, copy=False)
        Q = Q.astype(np.float32, copy=False)
        X_flat = X.reshape(X.shape[0], -1)
        Q_flat = Q.reshape(Q.shape[0], -1)

        # Preserve raw flattened vectors before L2-normalization so we can
        # compute distances in the original (raw) space later.
        X_flat_raw = X_flat.copy()
        Q_flat_raw = Q_flat.copy()

        # Guard against zero norms
        X_norms = np.linalg.norm(X_flat, axis=1, keepdims=True)
        X_norms[X_norms == 0] = 1.0
        Q_norms = np.linalg.norm(Q_flat, axis=1, keepdims=True)
        Q_norms[Q_norms == 0] = 1.0
        X_flat = X_flat / X_norms
        Q_flat = Q_flat / Q_norms
        # reshape back to (N,1,1,dim)
        X = X_flat.reshape(X.shape[0], 1, 1, -1)
        Q = Q_flat.reshape(Q.shape[0], 1, 1, -1)
    else:
        # image data

        # Keep raw flattened image vectors before scaling to [0,1]
        X_flat_raw = X.reshape(X.shape[0], -1).astype(np.float32, copy=True)
        Q_flat_raw = Q.reshape(Q.shape[0], -1).astype(np.float32, copy=True)

        X = X / 255.0
        Q = Q / 255.0

    
    print(f"Dataset shape: {X.shape}, Query shape: {Q.shape}")
    print(f"Attempting to find {args.N} neighbors using Neural LSH (T={args.T}).")

    # --- Neural LSH Search ---
    R = args.R
    range_enabled = (args.range.lower() == "true")

    results = []
    all_lsh_neighbors = []

    # --- Metrics containers ---
    all_lsh_neighbors: List[np.ndarray] = []
    all_AF = []
    all_t_approx = []
    all_t_true = []
    
    # --- Compute true neighbors once (normalized space) ---
    n_queries = Q.shape[0]
    if n_queries > 0:
        # true_neighbors_array, all_t_true = load_or_compute_true_neighbors(
        #     X, Q, args.dataset, args.query, args.N, true_neighbors_file=None, cache_dir="."
        # )
        from libraries import bruteforce_naive as bf_naive
        true_neighbors, t = bf_naive.load_or_compute_true_neighbors_naive(X, Q, 'dataset', 'query', args.N)
    else:
        true_neighbors_array = []

    if model and inverted:
        # Batch inference and vectorized re-ranking
        # Tune these if needed
        INFER_BATCH = 256
        CAND_CHUNK = 60000

        # Pre-flatten dataset (view when plibraries.ossible)
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32, copy=False)
        Q_flat_all = Q.reshape(Q.shape[0], -1).astype(np.float32, copy=False)

        # Keep references to the original/raw flattened arrays if we saved them
        X_flat_raw_all = X_flat_raw if 'X_flat_raw' in locals() else None
        Q_flat_raw_all = Q_flat_raw if 'Q_flat_raw' in locals() else None

        n_queries = Q.shape[0]
        print(libraries.time.strftime("%Y-%m-%d %H:%M:%S", libraries.time.localtime()), f"Processing {n_queries} queries in batches of {INFER_BATCH}...")
        for qi in range(0, n_queries, INFER_BATCH):
            q_slice = slice(qi, min(qi + INFER_BATCH, n_queries))
            q_batch = Q[q_slice]  # shape (b, 1, R, C) or (b,1,1,dim)
            b = q_batch.shape[0]

            # --- Start timing the whole LSH search for this batch ---
            t0_batch = libraries.time.perf_counter()
            print(libraries.time.strftime("%Y-%m-%d %H:%M:%S", libraries.time.localtime()), f"Processing batch starting at query index {qi}...")
            # Run model in batch
            q_tensor = libraries.torch.from_numpy(q_batch.astype(np.float32))
            with libraries.torch.no_grad():
                logits = model(q_tensor)
                if logits.ndim > 2:
                    logits = logits.reshape(logits.shape[0], -1)
                probs = libraries.torch.softmax(logits, dim=1).cpu().numpy()

            # For each query in batch, get top-T bins
            top_bins = np.argsort(probs, axis=1)[:, ::-1][:, :args.T]

            # Build per-batch union of candidate indices
            # We'll collect candidates per-query as well after getting the batch union
            batch_candidates = []
            for row in top_bins:
                row_candidates = []
                for bidx in row:
                    row_candidates.extend(inverted.get(int(bidx), []))
                batch_candidates.append(np.unique(row_candidates))

            # For vectorized re-ranking, also build the union to load chunks efficiently
            union_candidates = np.unique(np.concatenate(batch_candidates)) if len(batch_candidates) > 0 else np.array([], dtype=np.int64)

            # Precompute query norms (for normalized vectors used in ranking)
            Qb = Q_flat_all[q_slice]  # (b, d)
            Qb_norm_sq = np.sum(Qb * Qb, axis=1)[:, None]

            # If raw flattened vectors were preserved, prepare raw query norms
            Qb_raw = None
            Qb_raw_norm_sq = None
            if range_enabled and Q_flat_raw_all is not None:
                Qb_raw = Q_flat_raw_all[q_slice]
                Qb_raw_norm_sq = np.sum(Qb_raw * Qb_raw, axis=1)[:, None]

            # Prepare arrays to hold top-N for each query in batch
            top_idx = np.full((b, args.N), -1, dtype=np.int64)
            top_dist = np.full((b, args.N), np.inf, dtype=np.float32)
            # Prepare per-batch containers for range neighbors (R-near)
            batch_range_neighbors = [list() for _ in range(b)] if range_enabled else None

            if union_candidates.size > 0:
                # Process union_candidates in chunks to limit memory
                R2 = (R * R) if range_enabled else None
                for xi in range(0, union_candidates.size, CAND_CHUNK):
                    chunk = union_candidates[xi:xi + CAND_CHUNK]
                    # Normalized vectors used for ranking
                    Xc = X_flat[chunk]  # (chunk_size, d)
                    Xc_norm_sq = np.sum(Xc * Xc, axis=1)[None, :]

                    # compute distances (b, chunk_size) in normalized space for ranking
                    prod = Qb @ Xc.T
                    dist_sq = Qb_norm_sq - 2.0 * prod + Xc_norm_sq

                    # If range search enabled, prefer raw-space distance checks when
                    # raw flattened arrays are available (keeps R units consistent).
                    if range_enabled:
                        if X_flat_raw_all is not None and Qb_raw is not None:
                            Xc_raw = X_flat_raw_all[chunk]
                            Xc_raw_norm_sq = np.sum(Xc_raw * Xc_raw, axis=1)[None, :]
                            prod_raw = Qb_raw @ Xc_raw.T
                            dist_sq_raw = Qb_raw_norm_sq - 2.0 * prod_raw + Xc_raw_norm_sq
                            # For each query, find hits within raw R^2 and add global indices
                            for j in range(b):
                                hits = np.nonzero(dist_sq_raw[j] <= R2)[0]
                                if hits.size:
                                    batch_range_neighbors[j].extend((chunk[hits]).tolist())
                        else:
                            # Fall back to using the normalized-space distances for range tests
                            for j in range(b):
                                hits = np.nonzero(dist_sq[j] <= R2)[0]
                                if hits.size:
                                    batch_range_neighbors[j].extend((chunk[hits]).tolist())

                    # For each query, get local top-N within this chunk and merge
                    kth = min(args.N, dist_sq.shape[1] - 1)
                    # select up to args.N local candidates per query from this chunk
                    local_idx = np.argpartition(dist_sq, kth, axis=1)[:, :args.N]
                    rows = np.arange(dist_sq.shape[0])[:, None]
                    local_dists = dist_sq[rows, local_idx]
                    # map local indices to global dataset indices
                    global_indices = chunk[local_idx]

                    # Vectorized merge: combine existing top-N and local candidates, then select top-N
                    # cand_dists: (b, N + k), cand_idx: (b, N + k)
                    cand_dists = np.concatenate([top_dist, local_dists], axis=1)
                    cand_idx = np.concatenate([top_idx, global_indices], axis=1)
                    # Select indices of smallest args.N distances per row
                    part = np.argpartition(cand_dists, args.N, axis=1)[:, :args.N]
                    row_idx = np.arange(b)[:, None]
                    selected_dists = cand_dists[row_idx, part]
                    selected_idx = cand_idx[row_idx, part]
                    # Now sort selected distances within each row
                    order_within = np.argsort(selected_dists, axis=1)
                    top_dist = np.take_along_axis(selected_dists, order_within, axis=1)
                    top_idx = np.take_along_axis(selected_idx, order_within, axis=1)

            total_range_time = 0.0
            # For each query in batch, finalize neighbors and distances
            for local_i in range(b):
                q_global_i = qi + local_i
                # If no candidates, return empty
                if top_idx[local_i, 0] == -1:
                    neighbors = np.array([], dtype=int)
                    distances = np.array([], dtype=np.float32)
                else:
                    # top_idx contains dataset indices
                    neighbors = top_idx[local_i]
                    distances = np.sqrt(top_dist[local_i])
                all_lsh_neighbors.append(neighbors)

                # ---------- Approx distances: use stored top-N squared distances for normalized-space
                approx_dists_norm = np.sqrt(top_dist[local_i]) if top_dist.shape[1] > 0 else np.array([], dtype=np.float32)
                # Compute raw-space approximate distances only for the final top-N (if raw arrays present)
                if X_flat_raw_all is not None and Q_flat_raw_all is not None and neighbors.size > 0:
                    approx_dists_raw = np.linalg.norm(
                        X_flat_raw_all[neighbors] - Q_flat_raw_all[q_global_i], axis=1
                    )
                else:
                    approx_dists_raw = approx_dists_norm
                range_time_start = libraries.time.perf_counter()

                # If no approximate neighbors, record empty block and continue
                if neighbors.size == 0:
                    block = [f"Query: {q_global_i}", "(No neighbors)"]
                    results.append("\n".join(block))
                    continue

                # ---------- True distances timing (raw-space) ----------
                true_idx = true_neighbors_array[q_global_i]
                if X_flat_raw_all is not None and Q_flat_raw_all is not None:
                    true_dists = np.linalg.norm(
                        X_flat_raw_all[true_idx] - Q_flat_raw_all[q_global_i], axis=1
                    )
                else:
                    true_dists = np.linalg.norm(
                        X_flat[true_idx] - Q_flat_all[q_global_i], axis=1
                    )

                # ---------- Approximation factor ----------
                if true_dists.size > 0:
                    AF = sum(approx_dists_raw) / sum(true_dists)
                else:
                    AF = np.nan
                all_AF.append(AF)


                # Finalize range neighbors for this query (deduplicate and sort)
                if range_enabled and union_candidates.size > 0:
                    rr = np.unique(np.array(batch_range_neighbors[local_i], dtype=np.int64))
                else:
                    rr = np.array([], dtype=np.int64)

                # Use precomputed R-near lists collected during the LSH pass
                if range_enabled:
                    r_neighbors = rr if q_global_i < len(Q) else np.array([], dtype=np.int64)
                else:
                    r_neighbors = np.array([], dtype=np.int64)

                # ---------- Build output block ----------
                block = [f"Query: {q_global_i}"]
                for k in range(min(args.N, neighbors.size)):
                    block.append(f"Nearest neighbor-{k+1}: {int(neighbors[k])}")
                    block.append(f"distanceApproximate: {float(approx_dists_raw[k]):.4f}")
                    block.append(f"distanceTrue: {float(true_dists[k]):.4f}")

                if range_enabled and r_neighbors.size > 0:
                    block.append("R-near neighbors:")
                    for rid in r_neighbors:
                        block.append(str(int(rid)))

                results.append("\n".join(block))

                range_time_end = libraries.time.perf_counter()
                total_range_time += (range_time_end - range_time_start)

            # --- End timing ---
            t1_batch = libraries.time.perf_counter()
            t1_batch_real = t1_batch - total_range_time
            batch_time = t1_batch_real - t0_batch

            # Distribute batch_time equally across queries in the batch
            t_per_query = batch_time / b
            all_t_approx.extend([t_per_query] * b)

            print(f"Processed {min(qi + INFER_BATCH, n_queries)}/{n_queries} queries for LSH search ...")
    else:
        print("Skipping LSH search due to missing model or inverted file.")
        all_lsh_neighbors = [np.array([], dtype=int)] * len(Q)

    # ============================================================
    # Global metrics
    # ============================================================

    if all_AF:
        avg_AF = float(np.nanmean(all_AF))
    else:
        avg_AF = float('nan')

    if all_t_approx:
        tApprox_avg = float(np.mean(all_t_approx))
    else:
        tApprox_avg = 0.0

    if all_t_true:
        tTrue_avg = float(all_t_true / len(Q))
        print(f"Average true search time per query: {tTrue_avg:.6f} seconds")
        print(f"Total true search time for all queries: {all_t_true:.6f} seconds")
    else:
        tTrue_avg = 0.0

    if all_t_approx:
        total_approx_time = float(np.sum(all_t_approx))
    else:
        total_approx_time = 1e-9

    QPS = (len(Q) / total_approx_time) if total_approx_time > 0 else 0.0

    # Recall
    if n_queries > 0:
        recall_final = float(calculate_recall(true_neighbors_array, all_lsh_neighbors, args.N))
    else:
        recall_final = 0.0

    # ============================================================
    # Write output file
    # ============================================================

    with open(args.output, "w") as f:
        f.write("Neural LSH\n")

        for blk in results:
            f.write(blk + "\n\n")

        f.write(f"Average AF: {avg_AF:.4f}\n")
        f.write(f"Recall@{args.N}: {recall_final:.4f}\n")
        f.write(f"QPS: {QPS:.4f}\n")
        f.write(f"tApproximateAverage: {tApprox_avg:.6f}\n")
        f.write(f"tTrueAverage: {tTrue_avg:.6f}\n")

    print(f"Wrote output file to {args.output}")


if __name__ == "__main__":
    main()