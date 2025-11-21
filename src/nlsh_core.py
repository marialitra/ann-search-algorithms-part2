import libraries
from libraries import nn, np, List, Dict
from bruteforce import load_or_compute_true_neighbors, calculate_recall

def neural_lsh(args, model: nn.Module, inverted: Dict[int, List[int]],  X: np.ndarray, Q: np.ndarray, X_flat_raw: np.ndarray, Q_flat_raw: np.ndarray):
    
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
        true_neighbors_array, all_t_true = load_or_compute_true_neighbors( X, Q, args.dataset, args.query, args.N, true_neighbors_file=None, cache_dir=".")
    else:
        true_neighbors_array = []

    if model and inverted:
        # Batch inference and vectorized re-ranking
        INFER_BATCH = 256
        CAND_CHUNK = 60000

        # Pre-flatten dataset (view when plibraries.ossible)
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32, copy=False)
        Q_flat_all = Q.reshape(Q.shape[0], -1).astype(np.float32, copy=False)

        n_queries = Q.shape[0]
        for qi in range(0, n_queries, INFER_BATCH):
            q_slice = slice(qi, min(qi + INFER_BATCH, n_queries))
            q_batch = Q[q_slice]
            b = q_batch.shape[0]

            # Start timing the whole LSH search for this batch 
            t0_batch = libraries.time.perf_counter()

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
            batch_candidates = []
            for row in top_bins:
                row_candidates = []
                for bidx in row:
                    row_candidates.extend(inverted.get(int(bidx), []))
                batch_candidates.append(np.unique(row_candidates))

            # For vectorized re-ranking, also build the union to load chunks efficiently
            union_candidates = np.unique(np.concatenate(batch_candidates)) if len(batch_candidates) > 0 else np.array([], dtype=np.int64)

            # Precompute query norms for distance computations
            # Qb = Q_flat_all[q_slice]  # (b, d)
            Qb = Q_flat_all[q_slice].astype(np.float64, copy=False)
            # Qb_norm_sq = np.sum(Qb * Qb, axis=1)[:, None]
            Qb_norm_sq = np.sum(Qb * Qb, axis=1, dtype=np.float64)[:, None]

            # If raw flattened vectors were preserved, prepare raw query norms
            Qb_raw = None
            Qb_raw_norm_sq = None
            if range_enabled and Q_flat_raw is not None:
                Qb_raw = Q_flat_raw[q_slice]
                Qb_raw_norm_sq = np.sum(Qb_raw * Qb_raw, axis=1)[:, None]

            # Prepare arrays to hold top-N for each query in batch
            top_idx = np.full((b, args.N), -1, dtype=np.int64)
            # top_dist = np.full((b, args.N), np.inf, dtype=np.float32)
            top_dist = np.full((b, args.N), np.inf, dtype=np.float64)

            # Prepare per-batch containers for range neighbors (R-near)
            batch_range_neighbors = [list() for _ in range(b)] if range_enabled else None

            if union_candidates.size > 0:
                # Process union_candidates in chunks to limit memory
                R2 = (R * R) if range_enabled else None
                for xi in range(0, union_candidates.size, CAND_CHUNK):
                    chunk = union_candidates[xi:xi + CAND_CHUNK]
                    # Normalized vectors used for ranking
                    # Xc = X_flat[chunk]
                    Xc = X_flat[chunk].astype(np.float64, copy=False)
                    # Xc_norm_sq = np.sum(Xc * Xc, axis=1)[None, :]
                    Xc_norm_sq = np.sum(Xc * Xc, axis=1, dtype=np.float64)[None, :]

                    # Compute distances (b, chunk_size) in normalized space for ranking
                    prod = Qb @ Xc.T
                    dist_sq = Qb_norm_sq - 2.0 * prod + Xc_norm_sq # float64 now

                    # If range search enabled, prefer raw-space distance checks when
                    # Raw flattened arrays are available (keeps R units consistent).
                    if range_enabled:
                        if X_flat_raw is not None and Qb_raw is not None:
                            Xc_raw = X_flat_raw[chunk]
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

                    # Select up to args.N local candidates per query from this chunk
                    local_idx = np.argpartition(dist_sq, kth, axis=1)[:, :args.N]
                    rows = np.arange(dist_sq.shape[0])[:, None]
                    local_dists = dist_sq[rows, local_idx]

                    # Map local indices to global dataset indices
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
                if X_flat_raw is not None and Q_flat_raw is not None and neighbors.size > 0:
                    # approx_dists = np.linalg.norm(
                    #     X_flat_raw[neighbors] - Q_flat_raw[q_global_i], axis=1
                    # )
                    approx_dists = np.linalg.norm(
                        X_flat_raw[neighbors].astype(np.float64) - 
                        Q_flat_raw[q_global_i].astype(np.float64),
                        axis=1
                    ).astype(np.float64)
                else:
                    approx_dists = approx_dists_norm

                range_time_start = libraries.time.perf_counter()

                # If no approximate neighbors, record empty block and continue
                if neighbors.size == 0:
                    block = [f"Query: {q_global_i}", "(No neighbors)"]
                    results.append("\n".join(block))
                    continue

                # ---------- True distances timing (raw-space) ----------
                true_idx = true_neighbors_array[q_global_i]
                if X_flat_raw is not None and Q_flat_raw is not None:
                    # true_dists = np.linalg.norm(
                    #     X_flat_raw[true_idx] - Q_flat_raw[q_global_i], axis=1)
                    true_dists = np.linalg.norm(
                        X_flat_raw[true_idx].astype(np.float64) -
                        Q_flat_raw[q_global_i].astype(np.float64),
                        axis=1
                    ).astype(np.float64)
                else:
                    # true_dists = np.linalg.norm(
                    #     X_flat[true_idx] - Q_flat_all[q_global_i], axis=1)
                    true_dists = np.linalg.norm(
                        X_flat_raw[true_idx].astype(np.float64) -
                        Q_flat_raw[q_global_i].astype(np.float64),
                        axis=1
                    ).astype(np.float64)

                # ---------- Approximation factor ----------
                if true_dists.size > 0:
                    AF = sum(approx_dists) / sum(true_dists)
                else:
                    AF = np.nan
                all_AF.append(AF)


                # Finalize range neighbors for this query (deduplicate and sort)
                if range_enabled and union_candidates.size > 0:
                    rr = np.unique(np.array(batch_range_neighbors[local_i], dtype=np.int64))
                else:
                    rr = np.array([], dtype=np.int64)

                if range_enabled:
                    r_neighbors = rr if q_global_i < len(Q) else np.array([], dtype=np.int64)
                else:
                    r_neighbors = np.array([], dtype=np.int64)

                # ---------- Build output block ----------
                block = [f"Query: {q_global_i}"]
                for k in range(min(args.N, neighbors.size)):
                    block.append(f"Nearest neighbor-{k+1}: {int(neighbors[k])}")
                    block.append(f"distanceApproximate: {float(approx_dists[k]):.8f}")
                    block.append(f"distanceTrue: {float(true_dists[k]):.8f}")

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

    compute_metrics_produce_output(args, results, all_lsh_neighbors, true_neighbors_array, n_queries, all_AF, all_t_approx, all_t_true, Q)

    return

def compute_metrics_produce_output(args, results: List[str], all_lsh_neighbors: List[np.ndarray], true_neighbors_array: List[np.ndarray], n_queries: int, 
    all_AF: List[float], all_t_approx: List[float], all_t_true: float, Q: np.ndarray):
    # ---- Global metrics aggregation ----
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

    output_path = libraries.os.path.abspath(args.output)
    
    # ---- Write output file ----
    with open(output_path, "w") as f:
        f.write("Neural LSH\n")

        for blk in results:
            f.write(blk + "\n\n")

        f.write(f"Average AF: {avg_AF:.8f}\n")
        f.write(f"Recall@{args.N}: {recall_final:.8f}\n")
        f.write(f"QPS: {QPS:.8f}\n")
        f.write(f"tApproximateAverage: {tApprox_avg:.8f}\n")
        f.write(f"tTrueAverage: {tTrue_avg:.8f}\n")

    print(f"Wrote output file to {args.output}")
    return