import libraries
from libraries import Dict, List, Tuple, counter, np, nn, CNNClassifier, MLPClassifier, Optional, load_sift_vectors, load_idx_images

def validate_args(args):
    errors = []

    # Variables of build script
    if hasattr(args, "knn") and args.knn is not None:
        if args.knn <= 0:
            errors.append(f"--knn must be > 0, got {args.knn}")

    if hasattr(args, "m"):
        if args.m <= 0:
            errors.append(f"-m (number of blocks) must be > 0, got {args.m}")

    if hasattr(args, "imbalance"):
        if args.imbalance <= 0:
            errors.append(f"--imbalance must be > 0, got {args.imbalance}")

    if hasattr(args, "kahip_mode"):
        if args.kahip_mode not in (0, 1, 2):
            errors.append(f"--kahip_mode must be 0, 1, or 2, got {args.kahip_mode}")

    if hasattr(args, "layers"):
        if args.layers < 1:
            errors.append(f"--layers must be >= 1, got {args.layers}")

    if hasattr(args, "nodes"):
        if args.nodes <= 0:
            errors.append(f"--nodes must be > 0, got {args.nodes}")

    if hasattr(args, "epochs"):
        if args.epochs <= 0:
            errors.append(f"--epochs must be > 0, got {args.epochs}")

    if hasattr(args, "batch_size"):
        if args.batch_size <= 0:
            errors.append(f"--batch_size must be > 0, got {args.batch_size}")

    if hasattr(args, "lr"):
        if args.lr <= 0:
            errors.append(f"--lr (learning rate) must be > 0, got {args.lr}")

    # Variables of search script
    if hasattr(args, "N") and args.N is not None:
        if args.N <= 0:
            errors.append(f"-N (number of nearest neighbors) must be > 0, got {args.N}")

    if hasattr(args, "T") and args.T is not None:
        if args.T <= 0:
            errors.append(f"-T (number of bins to probe) must be > 0, got {args.T}")

    if errors:
        print("\nValidation errors:")
        for err in errors:
            print(f"  - {err}")
        print("\nPlease fix the arguments and try again.\n")
        raise SystemExit(1)


def _slug(s: str) -> str:
    """
        Make filesystem-safe short name from arbitrary string.
    """

    import re
    if s is None:
        return "unknown"

    s = str(s)
    s = s.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = s.strip('_')
    return s or "unknown"

def build_csr_from_neighbors(neighbors: Dict[int, List[int]], datasetsize: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # Build edge weight map
    array = {}
    for qid, nlist in neighbors.items():
        for nid in nlist:
            if qid == nid:
                continue
            array[(qid, nid)] = array.get((qid, nid), 0) + 1
            array[(nid, qid)] = array.get((nid, qid), 0) + 1
    sorted_values = dict(sorted(array.items(), key=lambda item: item[0][0]))
    
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

    adjncy = np.array(out_cols, dtype=np.int32)
    adjwgt = np.array(out_data, dtype=np.int32)
    vwgt = np.ones(datasetsize, dtype=np.int32)
    return xadj, adjncy, adjwgt, vwgt

def save_builds_output(model: nn.Module, out_dir: str, X: np.ndarray, y: np.ndarray, img_rows: int, img_cols: int):
    """
        Save model state_dict and inverted file (mapping block->indices).
    """
    libraries.os.makedirs(out_dir, exist_ok=True)
    model_path = libraries.os.path.join(out_dir, "model.pth")
    libraries.torch.save(model.state_dict(), model_path)
    print(f"Saved model state_dict to {model_path}")

    # Build inverted file
    inverted = {}
    unique_labels = np.unique(y)
    for label in unique_labels:
        inverted[int(label)] = np.where(y == label)[0].tolist()
    inv_path = libraries.os.path.join(out_dir, "inverted_file.npy")
    np.save(inv_path, inverted)
    print(f"Saved inverted file to {inv_path}")

    # Save simple metadata
    meta = {
        "n_vectors": int(X.shape[0]),
        "dim": int(X.shape[2] * X.shape[3]),
        "n_bins": int(len(unique_labels)),
        "img_rows": img_rows,
        "img_cols": img_cols
    }
    with open(libraries.os.path.join(out_dir, "meta.json"), "w") as fh:
        libraries.json.dump(meta, fh, indent=2)
    print("Saved meta.json")

def load_data(args) -> Tuple[nn.Module, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, List[int]]]:
    
    # Setup
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
        raise FileNotFoundError(f"Required meta.json file not found at {meta_path}")


    model_path = libraries.os.path.join(args.index, "model.pth")
    inverted_path = libraries.os.path.join(args.index, "inverted_file.npy")

    model = load_model(model_path, d_in, m, img_rows, img_cols)
    return model, *load_inverted_file(args, inverted_path, img_rows, img_cols)

def load_model(model_path: str, d_in: int, m: int, img_rows: int, img_cols: int) -> nn.Module:
    
    model = None
    if libraries.os.path.exists(model_path):
        # Load the checkpoint first and decide which classifier to instantiate
        sd = libraries.torch.load(model_path, map_location="cpu")
        
        # Handle common checkpoint wrappers where the state_dict is nested
        if isinstance(sd, dict):
            # Common keys used by training scripts
            for k in ("state_dict", "model_state_dict", "model", "net"): 
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break

        # Heuristic: if the saved state_dict contains 'conv1.0.weight' it's a CNN;
        # If it contains keys starting with 'net.' it's the MLP implementation.
        sd_keys = list(sd.keys()) if isinstance(sd, dict) else []

        if any(k.startswith("conv1") or k.startswith("conv1.0") for k in sd_keys):
            # CNN checkpoint: infer number of conv layers
            n_layers = 1 # count Final fully connected layer for classification
            for k in sd_keys: # look for conv layer weights
                v = sd[k]
                try :
                    shape = v.shape
                except Exception:
                    continue
                if len(shape) == 4:  # Conv layer weights are 4D tensors
                        n_layers += 1

            # print conv hiden layers and final out channels
            print(f"Loading CNN model with {n_layers} layers")
            model = CNNClassifier(img_rows=img_rows, img_cols=img_cols, n_out=m, n_layers=n_layers)
            model.load_state_dict(sd)
            model.eval()
        elif any(k.startswith("net.") for k in sd_keys):
            # MLP checkpoint: infer hidden size and number of linear layers
            # Find keys of the form 'net.<i>.weight'
            linear_keys = [k for k in sd_keys if k.startswith("net.") and k.endswith(".weight")]
            if not linear_keys:
                raise RuntimeError("Saved state_dict looks like an MLP but no linear weights found.")
            
            # Pick the first linear weight to infer hidden size and input dim
            first_w = sd[sorted(linear_keys)[0]]
            hidden_size = int(first_w.shape[0])

            # Number of linear layers == number of linear_keys
            n_linears = len(linear_keys)

            # Construct an MLP with that many linear layers
            model = MLPClassifier(d_in=d_in, n_out=m, hidden_size=hidden_size, n_layers=n_linears, dropout=0.0)
            model.load_state_dict(sd)
            model.eval()
        else:
            raise RuntimeError("Could not infer model architecture from saved state_dict. Please ensure the saved model matches the search-time model.")
    else:
        raise RuntimeError("Error: model.pth not found.")

    return model

def load_inverted_file(args, inverted_path: str, img_rows: int, img_cols: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, List[int]]]:

    inverted = {}
    is_sift = args.type and args.type.lower().startswith("sift")

    if libraries.os.path.exists(inverted_path):
        inverted = np.load(inverted_path, allow_pickle=True).item()
    else:
        raise RuntimeError("Error: inverted_file.npy not found.")

    # --- Load Data and Normalize ---
    if is_sift:
        # Load SIFT vectors
        X, num_images, num_rows, num_cols = load_sift_vectors(args.dataset)
        # Checks that everything is fine between the meta data and the actual data
        if img_rows != num_rows:
            raise ValueError(f"Invalid rows number in the images: {img_rows} != {num_rows}")
        
        if img_cols != num_cols:
            raise ValueError(f"Invalid columns number in the images: {img_cols} != {num_cols}")

        # Load query SIFT vectors 
        Q, num_images, num_rows, num_cols = load_sift_vectors(args.query)

        # Checks that everything is fine between the meta data and the actual data
        if img_rows != num_rows:
            raise ValueError(f"Invalid rows number in the query images: {img_rows} != {num_rows}")
        
        if img_cols != num_cols:
            raise ValueError(f"Invalid columns number in the query images: {img_cols} != {num_cols}")
    else:
        # Load IDX images (MNIST)
        X, num_images, num_rows, num_cols = load_idx_images(args.dataset)

        # Checks that everything is fine between the meta data and the actual data
        if img_rows != num_rows:
            raise ValueError(f"Invalid rows number in the images: {img_rows} != {num_rows}")
        
        if img_cols != num_cols:
            raise ValueError(f"Invalid columns number in the images: {img_cols} != {num_cols}")

        # Load query IDX images
        Q, num_images, num_rows, num_cols = load_idx_images(args.query)

        # Checks that everything is fine between the query data and the actual MNIST data
        if img_rows != num_rows:
            raise ValueError(f"Invalid rows number in the query images: {img_rows} != {num_rows}")
        
        if img_cols != num_cols:
            raise ValueError(f"Invalid columns number in the query images: {img_cols} != {num_cols}")

    return *normalize_data(X, Q, args), inverted

def normalize_data(X: np.ndarray, Q: np.ndarray, args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    # Normalize depending on dataset type.
    # - For image/IDX data (MNIST) scale 0-255 -> 0.0-1.0
    # - For SIFT (fvecs) do L2 normalization per-vector to match training preprocessing
    is_sift = args.type and args.type.lower().startswith("sift")

    if is_sift:
        X = X.astype(np.float32, copy=False)
        Q = Q.astype(np.float32, copy=False)
        X_flat = X.reshape(X.shape[0], -1)
        Q_flat = Q.reshape(Q.shape[0], -1)

        # Preserve raw flattened vectors before L2-normalization so we can
        # Compute distances in the original (raw) space later.
        X_flat_raw = X_flat.copy()
        Q_flat_raw = Q_flat.copy()

        # Guard against zero norms
        X_norms = np.linalg.norm(X_flat, axis=1, keepdims=True)
        X_norms[X_norms == 0] = 1.0
        Q_norms = np.linalg.norm(Q_flat, axis=1, keepdims=True)
        Q_norms[Q_norms == 0] = 1.0
        X_flat = X_flat / X_norms
        Q_flat = Q_flat / Q_norms
        
        # Reshape back to (N,1,1,dim)
        X = X_flat.reshape(X.shape[0], 1, 1, -1)
        Q = Q_flat.reshape(Q.shape[0], 1, 1, -1)
    else:
        # Keep raw flattened image vectors before scaling to [0,1]
        X_flat_raw = X.reshape(X.shape[0], -1).astype(np.float32, copy=True)
        Q_flat_raw = Q.reshape(Q.shape[0], -1).astype(np.float32, copy=True)

        X = X / 255.0
        Q = Q / 255.0

    return X, Q, X_flat_raw, Q_flat_raw