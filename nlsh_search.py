# neural with bruteforce and cnn handling
#!/usr/bin/env python3
import argparse
import os
import struct
import json
import numpy as np
import torch
# We renamed the classifier, but kept the old name for backward compatibility during import
# while updating the internal usage to the new CNN architecture.
from neural_net import CNNClassifier as Classifier 

from typing import Dict, List, Tuple


# --- Brute-Force and Caching Logic ---

def _slug(s: str) -> str:
    """Make filesystem-safe short name from arbitrary string."""
    import re
    if s is None:
        return "unknown"
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = s.strip('_')
    return s or "unknown"

def _make_cache_paths(base_dir: str, datasetName: str, queryName: str, N: int) -> Tuple[str, str]:
    """
    Returns (array_path, meta_path) for the cache files given dataset/query names and N.
    base_dir can be a folder; if empty, uses current folder.
    """
    base_dir = base_dir or "."
    os.makedirs(base_dir, exist_ok=True)
    fname = f"true_neighbors_{_slug(datasetName)}_{_slug(queryName)}_N{int(N)}.npy"
    meta = f"true_neighbors_{_slug(datasetName)}_{_slug(queryName)}_N{int(N)}.meta.json"
    return os.path.join(base_dir, fname), os.path.join(base_dir, meta)


def find_and_save_true_neighbors(X_flat: np.ndarray,
                                 Q_flat: np.ndarray,
                                 N: int,
                                 true_neighbors_file: str,
                                 datasetName: str,
                                 queryName: str) -> np.ndarray:
    """
    Compute brute-force top-N neighbors (exact) and save array and metadata.
    X_flat: (n, d), Q_flat: (nq, d)
    true_neighbors_file: path to .npy file (if None, will be auto-generated)
    datasetName, queryName: used for metadata & auto filename
    """
    # If user gave a directory or None, auto-generate filename in that directory
    if true_neighbors_file is None or os.path.isdir(true_neighbors_file):
        arr_path, meta_path = _make_cache_paths(true_neighbors_file, datasetName, queryName, N)
    else:
        # use given path; meta file placed next to it
        arr_path = true_neighbors_file
        meta_path = os.path.splitext(arr_path)[0] + ".meta.json"

    print("-" * 50)
    print(f"Brute-Force: Finding True Neighbors (N={N}) and Caching to '{arr_path}'...")
    total_queries = Q_flat.shape[0]

    # efficient squared distance computation (Q x X)
    X_norm_sq = np.sum(X_flat**2, axis=1)[np.newaxis, :]  # shape (1, n)
    Q_norm_sq = np.sum(Q_flat**2, axis=1)[:, np.newaxis]  # shape (nq, 1)
    dist_sq_matrix = Q_norm_sq - 2 * np.dot(Q_flat, X_flat.T) + X_norm_sq  # (nq, n)

    true_neighbors_indices = np.argsort(dist_sq_matrix, axis=1)[:, :N]  # (nq, N)

    # Save array and meta
    try:
        np.save(arr_path, true_neighbors_indices)
        meta = {
            "dataset_name": str(datasetName),
            "query_name": str(queryName),
            "N": int(N),
            "dataset_size": int(X_flat.shape[0]),
            "query_size": int(Q_flat.shape[0]),
            "dim": int(X_flat.shape[1])
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"Saved true neighbors to {arr_path}")
        print(f"Saved meta to {meta_path}")
    except Exception as e:
        print(f"Warning: could not save true neighbors or meta: {e}")

    print("-" * 50)
    return true_neighbors_indices


def load_or_compute_true_neighbors(X: np.ndarray,
                                   Q: np.ndarray,
                                   datasetName: str,
                                   queryName: str,
                                   N: int,
                                   true_neighbors_file: str = None,
                                   cache_dir: str = ".") -> np.ndarray:
    """
    Load cached true neighbors if compatible; otherwise compute and save.
    - If true_neighbors_file is provided and is a filepath it will be used.
    - If true_neighbors_file is None or points to a directory, the cache filename is auto-generated in cache_dir.
    """
    # Flatten inputs (safety)
    X_flat = X.reshape(X.shape[0], -1)
    Q_flat = Q.reshape(Q.shape[0], -1)

    # Determine paths
    if true_neighbors_file is None or os.path.isdir(true_neighbors_file):
        arr_path, meta_path = _make_cache_paths(cache_dir, datasetName, queryName, N)
    else:
        arr_path = true_neighbors_file
        meta_path = os.path.splitext(arr_path)[0] + ".meta.json"

    # If cache exists, validate meta
    if os.path.exists(arr_path) and os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path))
            # check N, sizes, names (you can be strict or lenient)
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

            # load array
            true_neighbors = np.load(arr_path)
            # sanity shape check
            if true_neighbors.ndim != 2 or true_neighbors.shape[0] != Q_flat.shape[0] or true_neighbors.shape[1] != N:
                print("Cached array shape mismatch; recomputing...")
                return find_and_save_true_neighbors(X_flat, Q_flat, N, arr_path, datasetName, queryName)

            print(f"Loaded cached true neighbors from {arr_path}")
            return true_neighbors

        except Exception as e:
            print(f"Error reading cache files ({e}); recomputing...")
            return find_and_save_true_neighbors(X_flat, Q_flat, N, arr_path, datasetName, queryName)
    else:
        # Cache not present
        return find_and_save_true_neighbors(X_flat, Q_flat, N, arr_path, datasetName, queryName)





# --- helpers to load datasets ---
def load_idx_images(filepath: str, img_rows: int = 0, img_cols: int = 0):
    """
    Load MNIST-like IDX images. 
    If img_rows/cols are provided, reshape to 4D (N, 1, R, C). 
    Otherwise, load and flatten (for initial metadata read if needed, though 'builder.py' handles this).
    """
    with open(filepath, 'rb') as f:
        header = f.read(16)
        magic, num_images, num_rows, num_cols = struct.unpack('>IIII', header)
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")
        
        # Use dimensions from arguments if available, otherwise use header info
        R = img_rows if img_rows else num_rows
        C = img_cols if img_cols else num_cols
        
        vector_dimension = num_rows * num_cols
        data_size = num_images * vector_dimension
        pixel_data = f.read(data_size)
        data = np.frombuffer(pixel_data, dtype=np.uint8)
        
        # Reshape to 4D for CNN: (N, C, H, W)
        data_vectors = data.reshape(num_images, 1, R, C).astype(np.float32)
        
    return data_vectors


def calculate_recall(true_neighbors_array: np.ndarray, all_lsh_neighbors: List[np.ndarray], N: int) -> float:
    """Calculates Recall@N given the true neighbors and the LSH results."""
    print("-" * 50)
    print(f"Starting Recall@{N} Evaluation...")
    
    total_queries = true_neighbors_array.shape[0]
    total_recall = 0.0

    if total_queries != len(all_lsh_neighbors):
        raise ValueError("Mismatch between true neighbor array size and LSH result size.")

    for qi in range(total_queries):
        # The true_neighbors_array is already a set of indices
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


def main():
    p = argparse.ArgumentParser(description="Neural LSH search phase.")
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
    meta_path = os.path.join(args.index_dir, "meta.json")
    
    # Read meta data, including new image dimensions
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        d_in = meta["dim"]
        m = meta["n_bins"]
        img_rows = meta.get("img_rows", 28) # Default to 28 for MNIST
        img_cols = meta.get("img_cols", 28) # Default to 28 for MNIST
    else:
        print("Warning: meta.json not found. Using defaults.")
        d_in, m, img_rows, img_cols = 784, 10, 28, 28
        
    model_path = os.path.join(args.index_dir, "model.pth")
    inverted_path = os.path.join(args.index_dir, "inverted_file.npy")
    
    model = None
    if os.path.exists(model_path):
        # Initialize the CNN model using image dimensions
        model = Classifier(img_rows=img_rows, img_cols=img_cols, n_out=m)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
    else:
        print("Warning: model.pth not found. Skipping model loading/hashing.")

    inverted = {}
    if os.path.exists(inverted_path):
        inverted = np.load(inverted_path, allow_pickle=True).item()
        print(f"Loaded inverted file with {len(inverted)} bins.")
    else:
        print("Warning: inverted_file.npy not found. Skipping LSH search.")

    # --- Load Data and Normalize ---
    print(f"Loading dataset and queries with shape (N, 1, {img_rows}, {img_cols}) ...")
    X = load_idx_images(args.dataset, img_rows, img_cols)
    Q = load_idx_images(args.query, img_rows, img_cols)

    # Normalize the pixel values (0-255) to float (0.0-1.0)
    X = X / 255.0 
    Q = Q / 255.0
    
    print(f"Dataset shape: {X.shape}, Query shape: {Q.shape}")
    print(f"Attempting to find {args.N} neighbors using Neural LSH (T={args.T}).")

    # --- Neural LSH Search ---
    results = []
    all_lsh_neighbors = [] 
    
    if model and inverted:
        for qi, q in enumerate(Q):
            # q is 4D (1, 1, R, C), unsqueeze(0) is not needed
            q_tensor = torch.from_numpy(q.astype(np.float32)).unsqueeze(0) 
            
            with torch.no_grad():
                logits = model(q_tensor)
                # Compute probabilities over the 'm' bins
                probs = torch.softmax(logits, dim=1).cpu().numpy().flatten() 

            # top-T bins are the most probable partitions
            top_bins = np.argsort(probs)[::-1][:args.T]
            candidates_idx = []
            for b in top_bins:
                candidates_idx.extend(inverted.get(int(b), []))
            candidates_idx = np.unique(candidates_idx)
            
            # Extract candidates and query vector, flattened for distance calculation
            # X and Q are 4D (N, 1, R, C), must flatten to 2D (N, D) for L2 norm
            candidates = X[candidates_idx].reshape(len(candidates_idx), -1)
            q_flat = q.reshape(-1)

            if len(candidates) > 0:
                # Calculate Euclidean distance between the query and all candidates
                dists = np.linalg.norm(candidates - q_flat, axis=1)
                topN_idx = np.argsort(dists)[:args.N]
                neighbors = candidates_idx[topN_idx]
                distances = dists[topN_idx]
            else:
                neighbors = np.array([], dtype=int)
                distances = np.array([], dtype=np.float32)

            all_lsh_neighbors.append(neighbors)

            result_lines = [f"Query: {qi}"]
            for j, (nid, dist) in enumerate(zip(neighbors, distances), 1):
                result_lines.append(f"Nearest neighbor-{j}: {nid}")
                result_lines.append(f"distanceApproximate: {dist:.4f}")
            results.append("\n".join(result_lines))

            if qi % 50 == 0:
                print(f"Processed {qi}/{len(Q)} queries for LSH search ...")
    
    else:
        print("Skipping LSH search due to missing model or inverted file.")
        all_lsh_neighbors = [np.array([], dtype=int)] * len(Q)


    # --- Evaluation ---
    if len(Q) > 0:
        # 1. Load or compute true neighbors (cached step)
        true_neighbors_array = load_or_compute_true_neighbors(
            X, Q, args.dataset, args.query, args.N, "true_neighbors.npy"
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