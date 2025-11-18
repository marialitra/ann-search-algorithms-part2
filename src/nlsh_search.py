import libraries
from libraries import np, CNNClassifier, MLPClassifier, Tuple, List, load_idx_images, load_sift_vectors

from bruteforce import load_or_compute_true_neighbors, calculate_recall

Classifier = None

def main():
    p = libraries.argparse.ArgumentParser(description="Neural LSH search phase.")
    p.add_argument("-d", "--dataset", required=True, help="dataset file (IDX or fvecs)")
    p.add_argument("-q", "--query", required=True, help="query file (IDX or fvecs)")
    p.add_argument("-i", "--index_dir", required=True, help="path to built index directory")
    p.add_argument("-o", "--output", required=True, help="output results file")
    p.add_argument("-type", help="type of the given dataset (MNIST or SIFT1M)")
    p.add_argument("-N", type=int, default=10, help="number of nearest neighbors to report")
    p.add_argument("-T", type=int, default=5, help="number of bins to probe (multi-probe)")
    p.add_argument("-range", type=str, default="false", help="range search flag")
    args = p.parse_args()

    # --- Setup ---
    print("Loading index ...")
    meta_path = libraries.os.path.join(args.index_dir, "meta.json")
    
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

    model_path = libraries.os.path.join(args.index_dir, "model.pth")
    inverted_path = libraries.os.path.join(args.index_dir, "inverted_file.npy")
    
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
        # 1. Load or compute true neighbors (cached step)
        # true_neighbors_array = load_or_compute_true_neighbors(
        #     X, Q, args.dataset, args.query, args.N, "true_neighbors.npy"
        # )
        
        true_neighbors_array = load_or_compute_true_neighbors(
            X, Q, args.dataset, args.query, args.N, true_neighbors_file=None, cache_dir="."
        )
        
        # 2. Calculate Recall
        calculate_recall(true_neighbors_array, all_lsh_neighbors, args.N)
    else:
        print("No queries to evaluate.")


    # write LSH results to output file
    with open(args.output, "w") as f:
        f.write("Neural LSH\n")
        f.write("\n\n".join(results))

    print(f"Search completed. LSH results written to {args.output}")
  
if __name__ == "__main__":
    main()