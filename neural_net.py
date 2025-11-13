import torch
import torch.nn as nn
import numpy as np

from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Define a simple feedforward neural network (MLP) 
# class MLPClassifier(nn.Module): 
#     def __init__(self, d_in, n_out, neurons, n_layers): 
#         super().__init__() 

#         layers = []
#         layers.append(nn.Linear(d_in, neurons))
#         layers.append(nn.ReLU())
#         for _ in range(n_layers - 2):
#             layers.append(nn.Linear(neurons, neurons))
#             layers.append(nn.ReLU())

#         layers.append(nn.Linear(neurons, n_out))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x): 
#         # Forward pass through the network 
#         return self.net(x)


# Define MLP
class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_out, hidden_size, n_layers, dropout):
        super().__init__()
        layers = []
        layers.append(nn.Linear(d_in, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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


def mnist_train(args, img_rows, img_cols, X, y):
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


def sift_train(args, img_rows, img_cols, X, y):

    """
    Lightweight and GPU-optimized training for large SIFT1M dataset.
    Uses a single train/val split instead of K-fold to reduce cost.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # SIFT X may be shaped (n,1,1,dim) coming from the loader. Flatten to (n, dim).
    X = X.reshape(X.shape[0], -1)

    # Normalize SIFT features (important!). Guard against zero-norm vectors.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = X / norms

    # Split train/val once
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # --- Configuration ---
    dropout_rate = getattr(args, "dropout_rate", 0.2)
    weight_decay = getattr(args, "weight_decay", 1e-5)
    patience = getattr(args, "patience", 3)

    model = MLPClassifier(
        d_in=X.shape[1],
        n_out=args.m,
        hidden_size=args.nodes,
        n_layers=args.layers,
        dropout=dropout_rate
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # DataLoaders
    # Create TensorDatasets from flattened arrays. Avoid unnecessary copies when possible.
    train_tensor_x = torch.from_numpy(X_train.copy()).float()
    val_tensor_x = torch.from_numpy(X_val.copy()).float()
    train_ds = torch.utils.data.TensorDataset(
        train_tensor_x,
        torch.from_numpy(y_train.copy()).long()
    )
    val_ds = torch.utils.data.TensorDataset(
        val_tensor_x,
        torch.from_numpy(y_val.copy()).long()
    )

    # Tune DataLoader for GPU if available
    use_cuda = (device.type == "cuda")
    dl_kwargs = dict(batch_size=args.batch_size, num_workers=4, pin_memory=use_cuda)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **dl_kwargs)

    # AMP scaler only needed on CUDA
    scaler = GradScaler() if use_cuda else None

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None


    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}.")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=use_cuda), yb.to(device, non_blocking=use_cuda)
            optimizer.zero_grad()

            if use_cuda:
                # enable AMP on CUDA devices
                with autocast(enabled=True):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                # backward with scaler if available
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # fallback if scaler not available
                    loss.backward()
                    optimizer.step()
            else:
                # CPU path: no AMP
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=use_cuda), yb.to(device, non_blocking=use_cuda)
                if use_cuda:
                    with autocast(enabled=True):
                        val_loss = loss_fn(model(xb), yb)
                else:
                    val_loss = loss_fn(model(xb), yb)
                total_val_loss += val_loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    print(f"Best val loss: {best_val_loss:.4f}")

    model = model.to("cpu")
    return model