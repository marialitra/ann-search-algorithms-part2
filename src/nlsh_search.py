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

    # --- Setup default R ---
    if args.R < 0:
        # Meaning that no given value has been given, or a wrong one detected,
        # We are going to use the defaut values for each dataset type

        if args.type and args.type.lower().startswith('sift'):
            args.R = 2800
        elif args.type and args.type.lower().startswith('mnist'):
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

        if any(k.startswith('conv1') or k.startswith('conv1.0') for k in sd_keys):
            # CNN checkpoint
            model = CNNClassifier(img_rows=img_rows, img_cols=img_cols, n_out=m)
            model.load_state_dict(sd)
            model.eval()
        elif any(k.startswith('net.') for k in sd_keys):
            # MLP checkpoint: infer hidden size and number of linear layers
            # find keys of the form 'net.<i>.weight'
            linear_keys = [k for k in sd_keys if k.startswith('net.') and k.endswith('.weight')]
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
    if args.type and args.type.lower().startswith('sift'):
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
        X = X / 255.0
        Q = Q / 255.0
    
    print(f"Dataset shape: {X.shape}, Query shape: {Q.shape}")
    print(f"Attempting to find {args.N} neighbors using Neural LSH (T={args.T}).")

    # --- Neural LSH Search ---
    results = []
    all_lsh_neighbors = [] 
    
    if model and inverted:
        # Batch inference and vectorized re-ranking
        # Tune these if needed
        INFER_BATCH = 128
        CAND_CHUNK = 10000

        # Pre-flatten dataset (view when plibraries.ossible)
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32, copy=False)
        Q_flat_all = Q.reshape(Q.shape[0], -1).astype(np.float32, copy=False)

        n_queries = Q.shape[0]
        for qi in range(0, n_queries, INFER_BATCH):
            q_slice = slice(qi, min(qi + INFER_BATCH, n_queries))
            q_batch = Q[q_slice]  # shape (b, 1, R, C) or (b,1,1,dim)
            b = q_batch.shape[0]

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

            # Precompute query norms
            Qb = Q_flat_all[q_slice]  # (b, d)
            Qb_norm_sq = np.sum(Qb * Qb, axis=1)[:, None]

            # Prepare arrays to hold top-N for each query in batch
            top_idx = np.full((b, args.N), -1, dtype=np.int64)
            top_dist = np.full((b, args.N), np.inf, dtype=np.float32)

            if union_candidates.size > 0:
                # Process union_candidates in chunks to limit memory
                for xi in range(0, union_candidates.size, CAND_CHUNK):
                    chunk = union_candidates[xi:xi + CAND_CHUNK]
                    Xc = X_flat[chunk]  # (chunk_size, d)
                    Xc_norm_sq = np.sum(Xc * Xc, axis=1)[None, :]

                    # compute distances (b, chunk_size)
                    prod = Qb @ Xc.T
                    dist_sq = Qb_norm_sq - 2.0 * prod + Xc_norm_sq

                    # For each query, get local top-N within this chunk and merge
                    kth = min(args.N, dist_sq.shape[1] - 1)
                    local_idx = np.argpartition(dist_sq, kth, axis=1)[:, :args.N]
                    rows = np.arange(dist_sq.shape[0])[:, None]
                    local_dists = dist_sq[rows, local_idx]
                    # map local indices to global dataset indices
                    global_indices = chunk[local_idx]

                    # Merge per-query
                    for j in range(b):
                        cand_dists = np.concatenate([top_dist[j], local_dists[j]])
                        cand_idx = np.concatenate([top_idx[j], global_indices[j]])
                        # pick top args.N
                        if cand_dists.size <= args.N:
                            order = np.argsort(cand_dists)
                        else:
                            order = np.argpartition(cand_dists, args.N)[:args.N]
                            order = order[np.argsort(cand_dists[order])]
                        top_dist[j] = cand_dists[order]
                        top_idx[j] = cand_idx[order]

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

                result_lines = [f"Query: {q_global_i}"]
                for j, (nid, dist) in enumerate(zip(neighbors, distances), 1):
                    result_lines.append(f"Nearest neighbor-{j}: {int(nid)}")
                    result_lines.append(f"distanceApproximate: {float(dist):.4f}")
                results.append("\n".join(result_lines))

            print(f"Processed {min(qi + INFER_BATCH, n_queries)}/{n_queries} queries for LSH search ...")
    
    else:
        print("Skipping LSH search due to missing model or inverted file.")
        all_lsh_neighbors = [np.array([], dtype=int)] * len(Q)


    # --- Evaluation ---
    if len(Q) > 0:
        true_neighbors_array = load_or_compute_true_neighbors(
            X, Q, args.dataset, args.query, args.N, true_neighbors_file=None, cache_dir="."
        )
        
        # 2. Calculate Recall
        calculate_recall(true_neighbors_array, all_lsh_neighbors, args.N)
    else:
        print("No queries to evaluate.")



    # ============================================================
    # Compute metrics per query
    # ============================================================

    all_AF = []                  # Approximation Factor per query
    all_range_neighbors = []     # Range search neighbors per query
    all_t_approx = []            # Approximate time per query
    all_t_true = []              # True time per query

    R = args.R if args.R > 0 else (2000 if args.type == "mnist" else 2800)
    range_enabled = (args.range.lower() == "true")

    print("Computing metrics for each query...")

    results = []

    for qi in range(len(all_lsh_neighbors)):

        approx_idx = all_lsh_neighbors[qi]                 # approximate neighbors
        true_idx = true_neighbors_array[qi]               # true neighbors

        if len(approx_idx) == 0 or approx_idx[0] < 0:
            # no neighbors case
            results.append(f"Query: {qi}\n(No neighbors)\n")
            continue

        # -------------------------------------------
        # Compute approximate distances with timing
        # -------------------------------------------
        t0 = libraries.time.perf_counter()
        approx_dists = np.linalg.norm(X_flat[approx_idx] - Q_flat_all[qi], axis=1)
        t1 = libraries.time.perf_counter()
        all_t_approx.append(t1 - t0)

        # -------------------------------------------
        # Compute true distances with timing
        # -------------------------------------------
        t0 = libraries.time.perf_counter()
        true_dists = np.linalg.norm(X_flat[true_idx] - Q_flat_all[qi], axis=1)
        t1 = libraries.time.perf_counter()
        all_t_true.append(t1 - t0)

        # -------------------------------------------
        # Approximation Factor (AF)
        # AF = d_approx(1) / d_true(1)
        # -------------------------------------------
        if true_dists[0] > 0:
            AF = approx_dists[0] / true_dists[0]
        else:
            AF = 1.0  # edge case
        all_AF.append(AF)

        # -------------------------------------------
        # Range Search R-near neighbors
        # -------------------------------------------
        if range_enabled:
            d_all = np.linalg.norm(X_flat - Q_flat_all[qi], axis=1)
            r_neighbors = np.where(d_all <= R)[0]
        else:
            r_neighbors = np.array([])
        all_range_neighbors.append(r_neighbors)

        # -------------------------------------------
        # Build formatted block for this query
        # -------------------------------------------

        block = []
        block.append(f"Query: {qi}")

        for k in range(args.N):
            block.append(f"Nearest neighbor-{k+1}: {int(approx_idx[k])}")
            block.append(f"distanceApproximate: {float(approx_dists[k]):.4f}")
            block.append(f"distanceTrue: {float(true_dists[k]):.4f}")

        if range_enabled:
            block.append("R-near neighbors:")
            for rid in r_neighbors:
                block.append(str(int(rid)))

        results.append("\n".join(block))


    # ============================================================
    # Compute global metrics
    # ============================================================

    avg_AF = float(np.mean(all_AF)) if all_AF else 0.0
    recall_final = float(calculate_recall(true_neighbors_array, all_lsh_neighbors, args.N))
    tApprox_avg = float(np.mean(all_t_approx)) if all_t_approx else 0.0
    tTrue_avg = float(np.mean(all_t_true)) if all_t_true else 0.0

    # QPS = queries per second for approximate search
    total_approx_time = sum(all_t_approx) if all_t_approx else 1e-9
    QPS = len(all_lsh_neighbors) / total_approx_time


    # ============================================================
    # Write final output file (EXACT FORMAT)
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