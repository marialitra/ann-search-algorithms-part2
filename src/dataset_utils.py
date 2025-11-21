import libraries
from libraries import np, nn, Sequence, Optional

class MemmapSIFTDataset(libraries.torch.utils.data.Dataset):
    """ Dataset that reads rows from a NumPy memmap or ndarray on demand.
        Returns (tensor_features, int_label) where tensor_features is float32
        and label is a torch.long.
    """
    def __init__(self, data: np.ndarray, labels: Sequence[int], indices: Optional[Sequence[int]] = None):
        # Data: memmap or ndarray shaped (n, dim) or (n,1,1,dim)
        self.data = data
        self.labels = labels
        if indices is None:
            self.indices = np.arange(len(labels), dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = int(self.indices[idx])

        # Read vector
        arr = self.data[i]

        # Flatten if necessary
        if arr.ndim > 1:
            arr = arr.reshape(-1)

        # Ensure float32 and L2-normalize the vector (guard zero-norm)
        a = np.asarray(arr, dtype=np.float32)
        norm = np.linalg.norm(a)
        if norm == 0:
            norm = 1.0
        a = a / norm

        # Convert to torch tensor and get label
        x = libraries.torch.from_numpy(a)
        y = int(self.labels[i])

        return x, libraries.torch.tensor(y, dtype=libraries.torch.long)

def make_sift_dataloaders(data: np.ndarray, labels: np.ndarray, train_idx, val_idx, batch_size: int = 256, num_workers: Optional[int] = None, pin_memory: bool = False, persistent_workers: bool = True):
    """
        Create DataLoaders for SIFT training using memmap-backed dataset.
        - data: np.ndarray or memmap shaped (n, ...) ; labels: 1D array-like
        - train_idx, val_idx: index arrays (numpy)
    """
    if num_workers is None:
        try:
            num_workers = max(0, min(8, (libraries.os.cpu_count() or 1) // 2))
        except Exception:
            num_workers = 4

    train_ds = MemmapSIFTDataset(data, labels, indices=train_idx)
    val_ds = MemmapSIFTDataset(data, labels, indices=val_idx)

    dl_kwargs = dict(batch_size=batch_size, shuffle=True, num_workers=num_workers,
                     pin_memory=pin_memory, persistent_workers=(persistent_workers and num_workers>0))

    train_loader = libraries.torch.utils.data.DataLoader(train_ds, **dl_kwargs)

    # validation loader shouldn't shuffle
    val_kwargs = dl_kwargs.copy()
    val_kwargs['shuffle'] = False
    val_loader = libraries.torch.utils.data.DataLoader(val_ds, **val_kwargs)

    return train_loader, val_loader