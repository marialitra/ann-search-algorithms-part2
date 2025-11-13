import torch
import torch.nn as nn

# Minimal, dependency-free replacement for sklearn.model_selection.train_test_split
# This avoids importing sklearn/pandas (which can trigger binary ABI issues
# with system-installed compiled packages). It supports `test_size` as float
# fraction or int count, `random_state` for reproducible shuffling, and an
# optional `stratify` array to preserve class proportions approximately.
import numpy as _np


def train_test_split(X, y, test_size=0.1, random_state=None, stratify=None):
    """Return (X_train, X_test, y_train, y_test) using NumPy only.

    Parameters:
    - X: array-like, shape (n_samples, ...)
    - y: array-like, shape (n_samples,)
    - test_size: float in (0,1) fraction or int number of test samples
    - random_state: int or None for RNG seed
    - stratify: array-like of labels to stratify by (optional)
    """
    X = _np.asarray(X)
    y = _np.asarray(y)
    n = X.shape[0]
    if isinstance(test_size, float):
        test_n = int(n * test_size)
    else:
        test_n = int(test_size)

    if random_state is not None:
        rng = _np.random.RandomState(random_state)
    else:
        rng = _np.random

    if stratify is None:
        idx = _np.arange(n)
        rng.shuffle(idx)
    else:
        labels = _np.asarray(stratify)
        uniq = _np.unique(labels)
        parts = []
        for u in uniq:
            ids = _np.where(labels == u)[0]
            # shuffle in-place using the RNG
            ids = ids.copy()
            rng.shuffle(ids)
            parts.append(ids)
        idx = _np.concatenate(parts) if parts else _np.arange(n)

    if test_n <= 0:
        return X, _np.empty((0,) + X.shape[1:]), y, _np.empty((0,), dtype=y.dtype)

    test_idx = idx[:test_n]
    train_idx = idx[test_n:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Define a simple feedforward neural network (MLP) 
class MLPClassifier(nn.Module): 
    def __init__(self, d_in, n_out, neurons, n_layers): 
        super().__init__() 

        layers = []
        layers.append(nn.Linear(d_in, neurons))
        layers.append(nn.ReLU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(neurons, neurons))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(neurons, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x): 
        # Forward pass through the network 
        return self.net(x)



# kfolds, regularization, dropout, cnn and pooling
import torch
import torch.nn as nn
import numpy as np

# Local KFold replacement to avoid depending on sklearn (prevents importing
# sklearn/pandas which can trigger binary incompatibilities in some systems).
class KFold:
    def __init__(self, n_splits=4, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(self):
        return self.n_splits

    def split(self, X):
        n = len(X)
        idx = np.arange(n)
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
        if self.shuffle:
            rng.shuffle(idx)
        fold_sizes = (n // self.n_splits) * np.ones(self.n_splits, dtype=int)
        remainder = n % self.n_splits
        for i in range(remainder):
            fold_sizes[i] += 1
        current = 0
        for fold_size in fold_sizes:
            start = current
            stop = current + fold_size
            test_idx = idx[start:stop]
            if start > 0 or stop < n:
                train_idx = np.concatenate([idx[:start], idx[stop:]])
            else:
                train_idx = np.array([], dtype=int)
            yield train_idx, test_idx
            current = stop

# --- CNN Classifier for Image Data ---
class CNNClassifier(nn.Module):
    def __init__(self, img_rows, img_cols, n_out, dropout_rate=0.25): 
        super().__init__()
        
        # Determine the size after the convolution and pooling layers
        # Simple two-layer CNN architecture suitable for MNIST (28x28)
        
        # Layer 1: Conv -> ReLU -> Pool -> Dropout
        self.conv1 = nn.Sequential(
            # Input: 1 channel (grayscale), Output: 32 feature maps
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces H/W by half (28x28 -> 14x14)
            nn.Dropout(dropout_rate)
        )
        
        # Layer 2: Conv -> ReLU -> Pool -> Dropout
        self.conv2 = nn.Sequential(
            # Input: 32 feature maps, Output: 64 feature maps
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces H/W by half (14x14 -> 7x7)
            nn.Dropout(dropout_rate)
        )
        
        # Calculate the size of the feature vector after convolution/pooling
        # If input is 28x28: After first pool: 14x14. After second pool: 7x7. 
        # Output channels from conv2 is 64.
        final_h = img_rows // 4 
        final_w = img_cols // 4
        fc_input_size = final_h * final_w * 64 
        
        print(f"CNN Feature Map size before Flatten: 64x{final_h}x{final_w}")
        print(f"Final Linear layer input size: {fc_input_size}")
        
        # Final fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, n_out) # n_out is the number of bins (args.m)
        )

    def forward(self, x): 
        x = self.conv1(x)
        x = self.conv2(x)
        return self.classifier(x)


def train_model(args, img_rows, img_cols, X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Configuration for Regularization ---
    dropout_rate = getattr(args, "dropout_rate", 0.25)
    weight_decay = getattr(args, "weight_decay", 1e-5) # L2 regularization

    kfold = KFold(n_splits=getattr(args, "kfolds", 4), shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print("-" * 60)
        print(f"Fold {fold+1}/{kfold.get_n_splits()}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Use the new CNNClassifier
        model = CNNClassifier(img_rows=img_rows, img_cols=img_cols, n_out=args.m,
                              dropout_rate=dropout_rate).to(device)
                              
        # Add weight_decay to the optimizer for L2 regularization
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        train_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train.copy()).float() / 255.0, # Normalize here
            torch.from_numpy(y_train.copy()).long())
        val_ds = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val.copy()).float() / 255.0, # Normalize here
            torch.from_numpy(y_val.copy()).long())

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 3
        best_model_state = None

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            for xd, yd in train_loader:
                xd, yd = xd.to(device), yd.to(device)
                optimizer.zero_grad()
                logits = model(xd)
                loss = loss_fn(logits, yd)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation loss
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xv, yv in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    val_loss += loss_fn(model(xv), yv).item()
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict() 
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        fold_results.append(best_val_loss)
        print(f"Fold {fold+1} best val loss: {best_val_loss:.4f}")

    print("-" * 60)
    print(f"Average validation loss across folds: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

    # Retrain on full dataset
    print("Retraining on full dataset with final configuration...")
    model = CNNClassifier(img_rows=img_rows, img_cols=img_cols, n_out=args.m,
                          dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    full_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X.copy()).float() / 255.0, # Normalize here
        torch.from_numpy(y.copy()).long())
    full_loader = torch.utils.data.DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xd, yd in full_loader:
            xd, yd = xd.to(device), yd.to(device)
            optimizer.zero_grad()
            logits = model(xd)
            loss = loss_fn(logits, yd)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"[Final Model] Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(full_loader):.4f}")

    model = model.to("cpu")
    return model