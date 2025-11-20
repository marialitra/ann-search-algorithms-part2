import libraries
from libraries import Dict, List, Tuple, counter, np, nn

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

    # build edge weight map
    array = {}
    for qid, nlist in neighbors.items():
        for nid in nlist:
            if qid == nid:
                continue
            array[(qid, nid)] = array.get((qid, nid), 0) + 1
            array[(nid, qid)] = array.get((nid, qid), 0) + 1
    sorted_values = dict(sorted(array.items(), key=lambda item: item[0][0]))
    # print(sorted_values)
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

def save_output(model: nn.Module, out_dir: str, X: np.ndarray, y: np.ndarray, img_rows: int, img_cols: int):
    """
        Save model state_dict and inverted file (mapping block->indices).
    """
    libraries.os.makedirs(out_dir, exist_ok=True)
    model_path = libraries.os.path.join(out_dir, "model.pth")
    libraries.torch.save(model.state_dict(), model_path)
    print(f"Saved model state_dict to {model_path}")

    # build inverted file
    inverted = {}
    unique_labels = np.unique(y)
    for label in unique_labels:
        inverted[int(label)] = np.where(y == label)[0].tolist()
    inv_path = libraries.os.path.join(out_dir, "inverted_file.npy")
    np.save(inv_path, inverted)
    print(f"Saved inverted file to {inv_path}")

    # save simple metadata
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