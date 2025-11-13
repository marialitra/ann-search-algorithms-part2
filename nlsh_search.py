# #!/usr/bin/env python3
# import argparse
# import os
# import struct
# import json
# import numpy as np
# import torch
# from neural_net import MLPClassifier

# # --- helpers to load datasets ---
# def load_idx_images(filepath: str):
#     """Load MNIST-like IDX images (as in build)."""
#     with open(filepath, 'rb') as f:
#         header = f.read(16)
#         magic, num_images, num_rows, num_cols = struct.unpack('>IIII', header)
#         if magic != 2051:
#             raise ValueError(f"Invalid magic number: {magic}")
#         vector_dimension = num_rows * num_cols
#         data_size = num_images * vector_dimension
#         pixel_data = f.read(data_size)
#         data = np.frombuffer(pixel_data, dtype=np.uint8)
#         data_vectors = data.reshape(num_images, vector_dimension).astype(np.float32)
#     return data_vectors

# # Euclidean distance
# def euclidean_distance(a, b):
#     # a: (n, d), b: (m, d)
#     return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)

# def main():
#     p = argparse.ArgumentParser(description="Neural LSH search phase.")
#     p.add_argument("-d", "--dataset", required=True, help="dataset file (IDX or fvecs)")
#     p.add_argument("-q", "--query", required=True, help="query file (IDX or fvecs)")
#     p.add_argument("-i", "--index_dir", required=True, help="path to built index directory")
#     p.add_argument("-o", "--output", required=True, help="output results file")
#     p.add_argument("-N", type=int, default=10, help="number of nearest neighbors to report")
#     p.add_argument("-T", type=int, default=5, help="number of bins to probe (multi-probe)")
#     p.add_argument("-range", type=str, default="false", help="range search flag")
#     args = p.parse_args()

#     print("Loading index ...")
#     meta_path = os.path.join(args.index_dir, "meta.json")
#     meta = json.load(open(meta_path))
#     d_in, m = meta["dim"], meta["n_bins"]

#     # Load model structure and weights
#     model = MLPClassifier(d_in, m, neurons=512, n_layers=3)
#     model.load_state_dict(torch.load(os.path.join(args.index_dir, "model.pth"), map_location="cpu"))
#     model.eval()

#     inverted = np.load(os.path.join(args.index_dir, "inverted_file.npy"), allow_pickle=True).item()
#     print(f"Loaded inverted file with {len(inverted)} bins.")

#     # Load dataset and query set
#     print("Loading dataset and queries ...")
#     X = load_idx_images(args.dataset)
#     Q = load_idx_images(args.query)

#     print(f"Dataset shape: {X.shape}, Query shape: {Q.shape}")

#     results = []

#     for qi, q in enumerate(Q):
#         q_tensor = torch.from_numpy(q.astype(np.float32)).unsqueeze(0)
#         with torch.no_grad():
#             logits = model(q_tensor)
#             probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

#         # top-T bins
#         top_bins = np.argsort(probs)[::-1][:args.T]
#         candidates_idx = []
#         for b in top_bins:
#             candidates_idx.extend(inverted.get(int(b), []))
#         candidates_idx = np.unique(candidates_idx)
#         candidates = X[candidates_idx]

#         # compute distances
#         dists = np.linalg.norm(candidates - q, axis=1)
#         topN_idx = np.argsort(dists)[:args.N]
#         neighbors = candidates_idx[topN_idx]
#         distances = dists[topN_idx]

#         # save per-query results
#         result_lines = [f"Query: {qi}"]
#         for j, (nid, dist) in enumerate(zip(neighbors, distances), 1):
#             result_lines.append(f"Nearest neighbor-{j}: {nid}")
#             result_lines.append(f"distanceApproximate: {dist:.4f}")
#         results.append("\n".join(result_lines))

#         if qi % 50 == 0:
#             print(f"Processed {qi}/{len(Q)} queries ...")

#     # write to output file
#     with open(args.output, "w") as f:
#         f.write("Neural LSH\n")
#         f.write("\n\n".join(results))

#     print(f"Search completed. Results written to {args.output}")

# if __name__ == "__main__":
#     main()


# neural with bruteforce saving

# #!/usr/bin/env python3
# import argparse
# import os
# import struct
# import json
# import numpy as np
# import torch
# from neural_net import MLPClassifier

# from typing import Dict, List, Tuple

# # --- helpers to load datasets ---
# def load_idx_images(filepath: str):
#     """Load MNIST-like IDX images (as in build)."""
#     # (Helper function remains the same as before)
#     with open(filepath, 'rb') as f:
#         header = f.read(16)
#         magic, num_images, num_rows, num_cols = struct.unpack('>IIII', header)
#         if magic != 2051:
#             raise ValueError(f"Invalid magic number: {magic}")
#         vector_dimension = num_rows * num_cols
#         data_size = num_images * vector_dimension
#         pixel_data = f.read(data_size)
#         data = np.frombuffer(pixel_data, dtype=np.uint8)
#         data_vectors = data.reshape(num_images, vector_dimension).astype(np.float32)
#     return data_vectors

# # --- Brute-Force and Caching Logic ---

# def find_and_save_true_neighbors(X: np.ndarray, Q: np.ndarray, N: int, true_neighbors_file: str) -> np.ndarray:
#     """
#     Performs brute-force search, saves the true neighbor indices, and returns them.
    
#     Returns:
#         A NumPy array of shape (n_query, N) containing the true neighbor indices.
#     """
#     print("-" * 50)
#     print(f"Brute-Force: Finding True Neighbors (N={N}) and Caching to '{true_neighbors_file}'...")
    
#     total_queries = Q.shape[0]
#     true_neighbors_indices = []

#     # Calculate squared Euclidean distance matrix (n_query x n_data)
#     # ||a-b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
#     X_norm_sq = np.sum(X**2, axis=1)[:, np.newaxis]
#     Q_norm_sq = np.sum(Q**2, axis=1)[:, np.newaxis]
#     dist_sq_matrix = Q_norm_sq - 2 * np.dot(Q, X.T) + X_norm_sq.T 
    
#     for qi in range(total_queries):
#         if qi % 100 == 0:
#             print(f"  ... processing query {qi}/{total_queries}")
            
#         # Get true top-N neighbor indices
#         true_topN_indices = np.argsort(dist_sq_matrix[qi])[:N]
#         true_neighbors_indices.append(true_topN_indices)
        
#     # Convert list of arrays to a single NumPy array (n_query, N)
#     true_neighbors_array = np.array(true_neighbors_indices)
    
#     # Save the results
#     try:
#         np.save(true_neighbors_file, true_neighbors_array)
#         print(f"Brute-Force: Successfully saved true neighbors to {true_neighbors_file}")
#     except Exception as e:
#         print(f"Warning: Could not save true neighbors file: {e}")

#     print("-" * 50)
#     return true_neighbors_array


# def load_or_compute_true_neighbors(X: np.ndarray, Q: np.ndarray, N: int, true_neighbors_file: str) -> np.ndarray:
#     """Loads true neighbors from file if available, otherwise computes and saves them."""
#     if os.path.exists(true_neighbors_file):
#         print("-" * 50)
#         print(f"Caching: Loading true neighbors from '{true_neighbors_file}'...")
#         try:
#             true_neighbors = np.load(true_neighbors_file)
#             if true_neighbors.shape[1] != N:
#                 print(f"Warning: Cached file has N={true_neighbors.shape[1]}, but requested N={N}. Recomputing...")
#                 return find_and_save_true_neighbors(X, Q, N, true_neighbors_file)
#             print("Caching: True neighbors loaded successfully.")
#             print("-" * 50)
#             return true_neighbors
#         except Exception as e:
#             print(f"Error loading cache file: {e}. Recomputing...")
#             return find_and_save_true_neighbors(X, Q, N, true_neighbors_file)
#     else:
#         return find_and_save_true_neighbors(X, Q, N, true_neighbors_file)


# def calculate_recall(true_neighbors_array: np.ndarray, all_lsh_neighbors: List[np.ndarray], N: int) -> float:
#     """Calculates Recall@N given the true neighbors and the LSH results."""
#     print("-" * 50)
#     print(f"Starting Recall@{N} Evaluation...")
    
#     total_queries = true_neighbors_array.shape[0]
#     total_recall = 0.0

#     if total_queries != len(all_lsh_neighbors):
#         raise ValueError("Mismatch between true neighbor array size and LSH result size.")

#     for qi in range(total_queries):
#         true_topN_indices = set(true_neighbors_array[qi])
#         lsh_neighbors = set(all_lsh_neighbors[qi])
        
#         # Intersection: True Positives found by LSH
#         hits = len(true_topN_indices.intersection(lsh_neighbors))
        
#         # Recall@N: (True Positives) / N
#         recall_q = hits / N
#         total_recall += recall_q
    
#     average_recall = total_recall / total_queries
    
#     print(f"Evaluation Completed. Average Recall@{N}: {average_recall:.4f}")
#     print("-" * 50)
    
#     return average_recall


# def main():
#     p = argparse.ArgumentParser(description="Neural LSH search phase.")
#     p.add_argument("-d", "--dataset", required=True, help="dataset file (IDX or fvecs)")
#     p.add_argument("-q", "--query", required=True, help="query file (IDX or fvecs)")
#     p.add_argument("-i", "--index_dir", required=True, help="path to built index directory")
#     p.add_argument("-o", "--output", required=True, help="output results file")
#     p.add_argument("-N", type=int, default=10, help="number of nearest neighbors to report")
#     p.add_argument("-T", type=int, default=5, help="number of bins to probe (multi-probe)")
#     p.add_argument("-range", type=str, default="false", help="range search flag")
#     p.add_argument("--true-neighbors-file", default="true_neighbors.npy", 
#                    help="Path to cache the true neighbor indices (Brute-Force result).")
#     args = p.parse_args()

#     # --- Setup ---
#     print("Loading index ...")
#     meta_path = os.path.join(args.index_dir, "meta.json")
    
#     if os.path.exists(meta_path):
#         meta = json.load(open(meta_path))
#         d_in, m = meta["dim"], meta["n_bins"]
#     else:
#         print("Warning: meta.json not found. Using defaults.")
#         d_in = 784 
#         m = 10 
        
#     model_path = os.path.join(args.index_dir, "model.pth")
#     inverted_path = os.path.join(args.index_dir, "inverted_file.npy")
    
#     model = None
#     if os.path.exists(model_path):
#         model = MLPClassifier(d_in, m, neurons=512, n_layers=3)
#         model.load_state_dict(torch.load(model_path, map_location="cpu"))
#         model.eval()
#     else:
#         print("Warning: model.pth not found. Skipping model loading/hashing.")

#     inverted = {}
#     if os.path.exists(inverted_path):
#         inverted = np.load(inverted_path, allow_pickle=True).item()
#         print(f"Loaded inverted file with {len(inverted)} bins.")
#     else:
#         print("Warning: inverted_file.npy not found. Skipping LSH search.")

#     # --- Load Data and Normalize ---
#     print("Loading dataset and queries ...")
#     X = load_idx_images(args.dataset)
#     Q = load_idx_images(args.query)

#     X = X / 255.0 
#     Q = Q / 255.0
    
#     print(f"Dataset shape: {X.shape}, Query shape: {Q.shape}")
#     print(f"Attempting to find {args.N} neighbors using LSH and Multi-Probe (T={args.T}).")

    
#     # --- Neural LSH Search ---
#     results = []
#     all_lsh_neighbors = [] 
    
#     if model and inverted:
#         for qi, q in enumerate(Q):
#             q_tensor = torch.from_numpy(q.astype(np.float32)).unsqueeze(0)
            
#             with torch.no_grad():
#                 logits = model(q_tensor)
#                 probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

#             top_bins = np.argsort(probs)[::-1][:args.T]
#             candidates_idx = []
#             for b in top_bins:
#                 candidates_idx.extend(inverted.get(int(b), []))
#             candidates_idx = np.unique(candidates_idx)
#             candidates = X[candidates_idx]

#             if len(candidates) > 0:
#                 dists = np.linalg.norm(candidates - q, axis=1)
#                 topN_idx = np.argsort(dists)[:args.N]
#                 neighbors = candidates_idx[topN_idx]
#                 distances = dists[topN_idx]
#             else:
#                 neighbors = np.array([], dtype=int)
#                 distances = np.array([], dtype=np.float32)

#             all_lsh_neighbors.append(neighbors)

#             result_lines = [f"Query: {qi}"]
#             for j, (nid, dist) in enumerate(zip(neighbors, distances), 1):
#                 result_lines.append(f"Nearest neighbor-{j}: {nid}")
#                 result_lines.append(f"distanceApproximate: {dist:.4f}")
#             results.append("\n".join(result_lines))

#             if qi % 50 == 0:
#                 print(f"Processed {qi}/{len(Q)} queries for LSH search ...")
    
#     else:
#         print("Skipping LSH search due to missing model or inverted file.")
#         all_lsh_neighbors = [np.array([], dtype=int)] * len(Q)


#     # --- Evaluation ---
#     if len(Q) > 0:
#         # 1. Load or compute true neighbors (cached step)
#         true_neighbors_array = load_or_compute_true_neighbors(
#             X, Q, args.N, args.true_neighbors_file
#         )
        
#         # 2. Calculate Recall
#         calculate_recall(true_neighbors_array, all_lsh_neighbors, args.N)
#     else:
#         print("No queries to evaluate.")


#     # write LSH results to output file
#     with open(args.output, "w") as f:
#         f.write("Neural LSH\n")
#         f.write("\n\n".join(results))

#     print(f"Search completed. LSH results written to {args.output}")

# if __name__ == "__main__":
#     main()




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

# --- Brute-Force and Caching Logic ---

def find_and_save_true_neighbors(X_flat: np.ndarray, Q_flat: np.ndarray, N: int, true_neighbors_file: str) -> np.ndarray:
    """
    Performs brute-force search using *flattened* data for distance calculation, 
    saves the true neighbor indices, and returns them.
    
    NOTE: X and Q must be passed as flattened (N, D) arrays for distance calculation.
    """
    print("-" * 50)
    print(f"Brute-Force: Finding True Neighbors (N={N}) and Caching to '{true_neighbors_file}'...")
    
    total_queries = Q_flat.shape[0]
    true_neighbors_indices = []

    # Calculate squared Euclidean distance matrix (n_query x n_data)
    # The data must be 2D (N, D) for this dot product to work.
    X_norm_sq = np.sum(X_flat**2, axis=1)[:, np.newaxis]
    Q_norm_sq = np.sum(Q_flat**2, axis=1)[:, np.newaxis]
    dist_sq_matrix = Q_norm_sq - 2 * np.dot(Q_flat, X_flat.T) + X_norm_sq.T 
    
    for qi in range(total_queries):
        if qi % 100 == 0 and qi > 0:
            print(f"  ... processing query {qi}/{total_queries}")
            
        # Get true top-N neighbor indices
        true_topN_indices = np.argsort(dist_sq_matrix[qi])[:N]
        true_neighbors_indices.append(true_topN_indices)
        
    # Convert list of arrays to a single NumPy array (n_query, N)
    true_neighbors_array = np.array(true_neighbors_indices)
    
    # Save the results
    try:
        np.save(true_neighbors_file, true_neighbors_array)
        print(f"Brute-Force: Successfully saved true neighbors to {true_neighbors_file}")
    except Exception as e:
        print(f"Warning: Could not save true neighbors file: {e}")

    print("-" * 50)
    return true_neighbors_array


def load_or_compute_true_neighbors(X: np.ndarray, Q: np.ndarray, N: int, true_neighbors_file: str) -> np.ndarray:
    """Loads true neighbors from file if available, otherwise computes and saves them."""
    
    # Flatten the 4D data back to 2D (N, D) for correct distance calculation
    X_flat = X.reshape(X.shape[0], -1)
    Q_flat = Q.reshape(Q.shape[0], -1)
    
    current_num_queries = Q_flat.shape[0]

    if os.path.exists(true_neighbors_file):
        print("-" * 50)
        print(f"Caching: Loading true neighbors from '{true_neighbors_file}'...")
        try:
            true_neighbors = np.load(true_neighbors_file)
            
            cached_num_queries = true_neighbors.shape[0]
            cached_N = true_neighbors.shape[1]
            
            recompute_reason = None
            if cached_N != N:
                recompute_reason = f"requested N={N} does not match cached N={cached_N}"
            elif cached_num_queries != current_num_queries:
                recompute_reason = f"current query set size ({current_num_queries}) does not match cached query size ({cached_num_queries})"
            
            if recompute_reason:
                print(f"Warning: Cached file is incompatible because {recompute_reason}. Recomputing...")
                return find_and_save_true_neighbors(X_flat, Q_flat, N, true_neighbors_file)
            
            print("Caching: True neighbors loaded successfully.")
            print("-" * 50)
            return true_neighbors
        except Exception as e:
            # This handles file corruption or non-NumPy data
            print(f"Error loading cache file: {e}. Recomputing...")
            return find_and_save_true_neighbors(X_flat, Q_flat, N, true_neighbors_file)
    else:
        # Cache file does not exist, compute and save
        return find_and_save_true_neighbors(X_flat, Q_flat, N, true_neighbors_file)


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
    p.add_argument("-N", type=int, default=10, help="number of nearest neighbors to report")
    p.add_argument("-T", type=int, default=5, help="number of bins to probe (multi-probe)")
    p.add_argument("-range", type=str, default="false", help="range search flag")
    p.add_argument("--true-neighbors-file", default="true_neighbors.npy", 
                    help="Path to cache the true neighbor indices (Brute-Force result).")
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
            X, Q, args.N, args.true_neighbors_file
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