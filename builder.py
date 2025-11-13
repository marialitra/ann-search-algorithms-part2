# import argparse
# import os
# import re
# import numpy as np
# import struct
# import json

# import kahip

# import torch
# import torch.nn as nn

# from neural_net import train_model

# from typing import Dict, List, Tuple
# from collections import Counter as counter


# def parse_neighbor_file(path: str) -> Tuple[Dict[int, List[int]], int]:
#     """Parse the neighbor file.

#     Returns:
#         neighbors: dict query -> list of neighbor ids (ints) in order
#     """
#     neighbors: Dict[int, List[int]] = {}

#     # regex helpers
#     q_re = re.compile(r"^Query:\s*(\d+)")
#     nn_re = re.compile(r"^Nearest neighbor-\d+:\s*(\d+)")

#     current_q = None
#     # temporary index to align distances
#     with open(path, "r", encoding="utf-8") as fh:
#         for raw in fh:
#             line = raw.strip()
#             if not line:
#                 continue
#             m = q_re.match(line)
#             if m:
#                 current_q = int(m.group(1))
#                 neighbors[current_q] = []
#                 continue

#             m = nn_re.match(line)
#             if m and current_q is not None:
#                 nid = int(m.group(1))
#                 neighbors[current_q].append(nid)
#                 continue

#     return neighbors, len(neighbors)


# def build_csr_from_neighbors(neighbors: Dict[int, List[int]], datasetsize: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

#     # build edge weight map
#     array = {}
#     for qid, nlist in neighbors.items():
#         for nid in nlist:
#             if qid == nid:
#                 continue
#             array[(qid, nid)] = array.get((qid, nid), 0) + 1
#             array[(nid, qid)] = array.get((nid, qid), 0) + 1
#     sorted_values = dict(sorted(array.items(), key=lambda item: item[0][0]))
#     print(sorted_values)
#     out_cols = []
#     out_data = []

#     for r, c in sorted_values.keys():
#         out_cols.append(c)
#         out_data.append(sorted_values[r, c])


#     rows = [k[0] for k in array.keys()]
#     counts = counter(rows)

#     xadj = np.zeros(datasetsize + 1, dtype=np.int32)

#     cum = 0
#     for i in range(datasetsize):
#         xadj[i] = cum 
#         cum += counts.get(i, 0)
#     xadj[datasetsize] = cum

#     # print("xadj:")
#     # print(xadj)

#     adjncy = np.array(out_cols, dtype=np.int32)
#     # print("adjncy")
#     # print(adjncy)
#     adjwgt = np.array(out_data, dtype=np.int32)
#     # print("adjwgt")
#     # print(adjwgt)
#     vwgt = np.ones(datasetsize, dtype=np.int32)
#     # print("vwgt")
#     # print(vwgt)
#     return xadj, adjncy, adjwgt, vwgt

# def load_idx_images(filepath: str) -> Tuple[np.ndarray, int, int]:
#     """
#     Reads and extracts image vectors from a file in the IDX format (like MNIST).

#     The IDX-3 header structure is:
#     - Magic number (4 bytes, integer, big-endian)
#     - Number of images (4 bytes, integer, big-endian)
#     - Number of rows (4 bytes, integer, big-endian)
#     - Number of columns (4 bytes, integer, big-endian)

#     Args:
#         filepath: The path to the IDX file (e.g., 'train-images-idx3-ubyte').

#     Returns:
#         A tuple containing:
#         - data_vectors: A NumPy array of shape (num_images, rows * cols), 
#           where each row is a flattened image vector (0-255).
#         - num_images: The total number of images found.
#         - vector_dimension: The dimension of the flattened vector (rows * cols).
#     """
#     print(f"Loading IDX image file: {filepath}")

#     if not os.path.exists(filepath):
#         raise FileNotFoundError(f"Error: File not found at {filepath}")

#     # 'rb' opens the file in binary read mode
#     with open(filepath, 'rb') as f:
#         # Read the first 4 integers (4 bytes each, big-endian/network byte order)
#         # '>iiii' means: Big-endian, four 4-byte signed integers
        
#         # Unpack the header: Magic Number, Num Images, Rows, Cols
#         header = f.read(16)
#         magic, num_images, num_rows, num_cols = struct.unpack('>IIII', header)
        
#         # Verify the magic number for IDX-3 (images) which should be 2051
#         if magic != 2051:
#             raise ValueError(f"Invalid magic number in IDX file header: {magic}. Expected 2051.")

#         print(f"Found {num_images} images, each {num_rows}x{num_cols}.")
#         vector_dimension = num_rows * num_cols
#         print(f"With dimension: {vector_dimension}.")

#         # Calculate the total size of the pixel data to read (bytes)
#         data_size = num_images * vector_dimension
        
#         # Read the rest of the file contents (the pixel data)
#         pixel_data = f.read(data_size)

#         # Convert the byte data into a NumPy array of unsigned 8-bit integers (0-255)
#         raw_data = np.frombuffer(pixel_data, dtype=np.uint8)

#         # Reshape the 1D array into a 2D array of (num_images, rows * cols)
#         data_vectors = raw_data.reshape(num_images, vector_dimension)

#     print("Successfully extracted vectors.")
#     return data_vectors, num_images, vector_dimension


# def save_output(model: nn.Module, out_dir: str, X: np.ndarray, y: np.ndarray):
#     """
#     Save model state_dict and inverted file (mapping block->indices).
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     model_path = os.path.join(out_dir, "model.pth")
#     torch.save(model.state_dict(), model_path)
#     print(f"Saved model state_dict to {model_path}")

#     # build inverted file
#     inverted = {}
#     unique_labels = np.unique(y)
#     for label in unique_labels:
#         inverted[int(label)] = np.where(y == label)[0].tolist()
#     inv_path = os.path.join(out_dir, "inverted_file.npy")
#     np.save(inv_path, inverted)
#     print(f"Saved inverted file to {inv_path}")

#     # save simple metadata
#     meta = {
#         "n_vectors": int(X.shape[0]),
#         "dim": int(X.shape[1]),
#         "n_bins": int(len(unique_labels))
#     }
#     with open(os.path.join(out_dir, "meta.json"), "w") as fh:
#         json.dump(meta, fh, indent=2)
#     print("Saved meta.json")


# def main():
#     p = argparse.ArgumentParser(description="Build adjacency matrix (CSR) from neighbor TXT files.")
#     p.add_argument("input", type=str, help="path to neighbor TXT file")
#     p.add_argument("-d", type=str, help="path to dataset file")
#     p.add_argument("-i", type=str, help="path to index file")
#     p.add_argument("--type", type=str, help="dataset type (MNIST or SIFT)")
#     p.add_argument("--nodes", type=int, help="number of neurons in layers")
#     p.add_argument("--layers", type=int, help="number of layers")
#     p.add_argument("--lr", type=float, help="learning rate")
#     p.add_argument("-m", type=int, help="number of blocks")
#     p.add_argument("--epochs", type=int, help="number of epochs in training loop")
#     p.add_argument("--batch_size", type=int, help="batch size")
#     args = p.parse_args()

#     inp = args.input
#     if not os.path.exists(inp):
#         raise SystemExit(f"File not found: {inp}")

#     dataset_type = args.type
#     print(f"The specified dataset type is: {dataset_type}")

#     data_vectors, num_images, vector_dimension = load_idx_images(args.d)

#     neighbors, datasetsize = parse_neighbor_file(inp)
#     print(f"Parsed {datasetsize} queries. Building adjacency...")

#     xadj, adjncy, adjwgt, vwgt = build_csr_from_neighbors(neighbors, datasetsize)

#     IMBALANCE = 0.03
#     SEED = 42

#     # Call kahip partitioner
#     edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjwgt, adjncy, args.m, IMBALANCE, True, SEED, 1)
#     print(f"Partitioned graph into {args.m} blocks with edgecut {edgecut}.")
#     print(blocks)

#     X = data_vectors  # shape (n, d)
#     y = np.array(blocks)  # labels from KaHIP
    
#     # for idx in range(num_images):
#     #     print(X[idx])
        
#     model = train_model(args, vector_dimension, X, y)
#     os.makedirs(args.i, exist_ok=True)

#     # Save the model and the inverted index file
#     save_output(model, args.i, X.copy(), y.copy())


# if __name__ == "__main__":
#     main()




# builder for cnn
import argparse
import os
import re
import numpy as np
import struct
import json

import kahip

import torch
import torch.nn as nn

from neural_net import train_model

from typing import Dict, List, Tuple
from collections import Counter as counter


def parse_neighbor_file(path: str) -> Tuple[Dict[int, List[int]], int]:
    """Parse the neighbor file.

    Returns:
        neighbors: dict query -> list of neighbor ids (ints) in order
    """
    neighbors: Dict[int, List[int]] = {}

    # regex helpers
    q_re = re.compile(r"^Query:\s*(\d+)")
    nn_re = re.compile(r"^Nearest neighbor-\d+:\s*(\d+)")

    current_q = None
    # temporary index to align distances
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            m = q_re.match(line)
            if m:
                current_q = int(m.group(1))
                neighbors[current_q] = []
                continue

            m = nn_re.match(line)
            if m and current_q is not None:
                nid = int(m.group(1))
                neighbors[current_q].append(nid)
                continue

    return neighbors, len(neighbors)


def build_csr_from_neighbors(neighbors: Dict[int, List[int]], datasetsize: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # build edge weight map
    array = {}
    for qid, nlist in neighbors.items():
        for nid in nlist:
            if qid == nid:
                continue
            array[(qid, nid)] = array.get((qid, nid), 0) + 1
            array[(nid, qid)] = array.get((nid, qid), 0) + 1
    sorted_values = dict(sorted(array.items(), key=lambda item: item[0][0]))
    print(sorted_values)
    out_cols = []
    out_data = []

    for r, c in sorted_values.keys():
        out_cols.append(c)
        out_data.append(sorted_values[r, c])


    rows = [k[0] for k in array.keys()]
    counts = counter(rows)

    xadj = np.zeros(datasetsize + 1, dtype=np.int32)

    cum = 0
    for i in range(datasetsize):
        xadj[i] = cum 
        cum += counts.get(i, 0)
    xadj[datasetsize] = cum

    # print("xadj:")
    # print(xadj)

    adjncy = np.array(out_cols, dtype=np.int32)
    # print("adjncy")
    # print(adjncy)
    adjwgt = np.array(out_data, dtype=np.int32)
    # print("adjwgt")
    # print(adjwgt)
    vwgt = np.ones(datasetsize, dtype=np.int32)
    # print("vwgt")
    # print(vwgt)
    return xadj, adjncy, adjwgt, vwgt

def load_idx_images(filepath: str) -> Tuple[np.ndarray, int, int, int]:
    """
    Reads and extracts image vectors from a file in the IDX format (like MNIST).
    
    Returns:
        A tuple containing:
        - data_vectors: A NumPy array of shape (num_images, 1, rows, cols), 
          where 1 is the channel dimension for grayscale (0-255).
        - num_images: The total number of images found.
        - num_rows: The height of the image.
        - num_cols: The width of the image.
    """
    print(f"Loading IDX image file: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: File not found at {filepath}")

    with open(filepath, 'rb') as f:
        header = f.read(16)
        # Unpack the header: Magic Number, Num Images, Rows, Cols
        magic, num_images, num_rows, num_cols = struct.unpack('>IIII', header)
        
        # Verify the magic number for IDX-3 (images) which should be 2051
        if magic != 2051:
            raise ValueError(f"Invalid magic number in IDX file header: {magic}. Expected 2051.")

        print(f"Found {num_images} images, each {num_rows}x{num_cols}.")
        vector_dimension = num_rows * num_cols
        print(f"Original flat dimension: {vector_dimension}.")

        # Calculate the total size of the pixel data to read (bytes)
        data_size = num_images * vector_dimension
        
        # Read the rest of the file contents (the pixel data)
        pixel_data = f.read(data_size)

        # Convert the byte data into a NumPy array of unsigned 8-bit integers (0-255)
        raw_data = np.frombuffer(pixel_data, dtype=np.uint8)

        # --- CRITICAL CHANGE: Reshape for CNN (Batch, Channels, Height, Width) ---
        # Channels=1 for grayscale
        data_vectors = raw_data.reshape(num_images, 1, num_rows, num_cols)

    print("Successfully extracted vectors.")
    return data_vectors, num_images, num_rows, num_cols


def save_output(model: nn.Module, out_dir: str, X: np.ndarray, y: np.ndarray, img_rows: int, img_cols: int):
    """
    Save model state_dict and inverted file (mapping block->indices).
    """
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model state_dict to {model_path}")

    # build inverted file
    inverted = {}
    unique_labels = np.unique(y)
    for label in unique_labels:
        inverted[int(label)] = np.where(y == label)[0].tolist()
    inv_path = os.path.join(out_dir, "inverted_file.npy")
    np.save(inv_path, inverted)
    print(f"Saved inverted file to {inv_path}")

    # save simple metadata
    meta = {
        "n_vectors": int(X.shape[0]),
        "dim": int(X.shape[2] * X.shape[3]), # Store the flat dimension for reference
        "n_bins": int(len(unique_labels)),
        "img_rows": img_rows, # Store dimensions for CNN
        "img_cols": img_cols  # Store dimensions for CNN
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print("Saved meta.json")


def main():
    p = argparse.ArgumentParser(description="Build adjacency matrix (CSR) from neighbor TXT files.")
    p.add_argument("input", type=str, help="path to neighbor TXT file")
    p.add_argument("-d", type=str, help="path to dataset file")
    p.add_argument("-i", type=str, help="path to index file")
    p.add_argument("--type", type=str, help="dataset type (MNIST or SIFT)")
    p.add_argument("--nodes", type=int, default=128, help="number of neurons in FC layers (less important for CNN)")
    p.add_argument("--layers", type=int, default=3, help="number of layers (less important for CNN)")
    p.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    p.add_argument("-m", type=int, required=True, help="number of blocks")
    p.add_argument("--epochs", type=int, default=10, help="number of epochs in training loop")
    p.add_argument("--batch_size", type=int, default=64, help="batch size")
    # Added new regularization parameters for potential command-line tuning
    p.add_argument("--dropout_rate", type=float, default=0.25, help="Dropout rate for CNN layers")
    p.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization for optimizer")
    p.add_argument("--kfolds", type=int, default=4, help="Number of folds for cross-validation")
    args = p.parse_args()

    inp = args.input
    if not os.path.exists(inp):
        raise SystemExit(f"File not found: {inp}")

    dataset_type = args.type
    print(f"The specified dataset type is: {dataset_type}")

    # Load 4D data (N, C, H, W)
    data_vectors, num_images, img_rows, img_cols = load_idx_images(args.d)

    neighbors, datasetsize = parse_neighbor_file(inp)
    print(f"Parsed {datasetsize} queries. Building adjacency...")

    xadj, adjncy, adjwgt, vwgt = build_csr_from_neighbors(neighbors, datasetsize)

    IMBALANCE = 0.03
    SEED = 42

    # Call kahip partitioner
    edgecut, blocks = kahip.kaffpa(vwgt, xadj, adjwgt, adjncy, args.m, IMBALANCE, True, SEED, 1)
    print(f"Partitioned graph into {args.m} blocks with edgecut {edgecut}.")
    print(blocks)

    X = data_vectors    # shape (n, 1, rows, cols)
    y = np.array(blocks)  # labels from KaHIP
    
    # Train the new CNN model, passing image dimensions
    model = train_model(args, img_rows, img_cols, X, y)
    os.makedirs(args.i, exist_ok=True)

    # Save the model and the inverted index file
    save_output(model, args.i, X.copy(), y.copy(), img_rows, img_cols)


if __name__ == "__main__":
    main()