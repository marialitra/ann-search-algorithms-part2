import libraries
from libraries import nn, np, train_test_split, GradScaler, autocast, make_sift_dataloaders

# Define MLP Classifier for SIFT data
class MLPClassifier(nn.Module):
    def __init__(self, d_in, n_out, hidden_size, n_layers, dropout):
        super().__init__()
        layers = []
        layers.append(nn.Linear(d_in, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        if n_layers > 0  and n_layers - 2 > 0 :
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
    def __init__(self, img_rows, img_cols, n_out, n_layers, dropout_rate=0.25): 
        super().__init__()

        # Layer 1: Conv -> ReLU -> Pool -> Dropout
        self.conv1 = nn.Sequential(
            # Input: 1 channel (grayscale), Output: 32 feature maps
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Reduces H/W by half (28x28 -> 14x14)
            nn.Dropout(dropout_rate)
        )
        # Build optional hidden conv blocks. We interpret `n_layers` such that
        # conv1 and conv2 are included in the count; the number of extra
        # convolutional blocks between them is `max(0, n_layers - 3)`.
        in_ch = 32
        out_ch = int(in_ch * 1.25)
        layers = []
        num_extra = max(0, n_layers - 3)
        for _ in range(num_extra):
            layers.append(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_ch = out_ch
            out_ch = min(int(out_ch * 1.2), 1024)

        print(f"CNN hidden conv layers: {num_extra}, final channels before conv2: {in_ch}")
        self.hidden = nn.Sequential(*layers)

        # conv2 consumes the channels produced by the hidden stack and performs
        # another pooling to reduce spatial dims to 7x7 for 28x28 input.
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            nn.Dropout(dropout_rate),
        )
    
        print(f"CNN conv2 output channels: {out_ch}")

        # Calculate the size of the feature vector after convolution/pooling
        final_h = img_rows // 4
        final_w = img_cols // 4
        fc_input_size = final_h * final_w * out_ch

        # Debuggine Reasons
        # print(f"CNN Feature Map size before Flatten: 64x{final_h}x{final_w}")
        # print(f"Final Linear layer input size: {fc_input_size}")
        
        # Final fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, n_out) # n_out is the number of bins (args.m)
        )

    def forward(self, x): 
        x = self.conv1(x)
        x = self.hidden(x)
        x = self.conv2(x)
        return self.classifier(x)


def mnist_train(args, img_rows, img_cols, X, y):
    """
        Lightweight and GPU-optimized training for MNIST-like datasets.
    """
    device = libraries.torch.device("cuda" if libraries.torch.cuda.is_available() else "cpu")
    
    # --- Configuration for Regularization ---
    dropout_rate = getattr(args, "dropout_rate", 0.25)
    weight_decay = getattr(args, "weight_decay", 1e-5) # L2 regularization

    # Normalize and split without copying large arrays where possible
    X_flat = X.reshape(X.shape[0], -1)
    indices = np.arange(X_flat.shape[0], dtype=np.int64)
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    # Build datasets/loaders
    train_ds = libraries.torch.utils.data.TensorDataset(
        libraries.torch.from_numpy((X[train_idx].astype(np.float32) / 255.0).copy()),
        libraries.torch.from_numpy(y[train_idx].copy()).long()
    )
    val_ds = libraries.torch.utils.data.TensorDataset(
        libraries.torch.from_numpy((X[val_idx].astype(np.float32) / 255.0).copy()),
        libraries.torch.from_numpy(y[val_idx].copy()).long()
    )

    train_loader = libraries.torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = libraries.torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)

    # Instantiate model and optimizer
    model = CNNClassifier(img_rows=img_rows, img_cols=img_cols, n_out=args.m, n_layers=args.layers,
                          dropout_rate=dropout_rate).to(device)
    optimizer = libraries.torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

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

        # Validation
        model.eval()
        val_loss = 0.0
        with libraries.torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                val_loss += loss_fn(model(xv), yv).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best val loss: {best_val_loss:.4f}")
    model = model.to("cpu")
    return model

def sift_train(args, img_rows, img_cols, X, y):

    """
        Lightweight and GPU-optimized training for large SIFT1M dataset.
    """

    device = libraries.torch.device("cuda" if libraries.torch.cuda.is_available() else "cpu")

    X = X.reshape(X.shape[0], -1)

    # Split train/val once 
    indices = np.arange(X.shape[0], dtype=np.int64)
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

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

    optimizer = libraries.torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # Create memmap-backed DataLoaders (no full-array copies)
    use_cuda = (device.type == "cuda")
    num_workers = getattr(args, "num_workers", None)
    pin_memory = use_cuda  # pin_memory only useful when transferring to CUDA
    train_loader, val_loader = make_sift_dataloaders(X, np.asarray(y), train_idx, val_idx,
                                                    batch_size=getattr(args, "batch_size", 256),
                                                    num_workers=num_workers,
                                                    pin_memory=pin_memory,
                                                    persistent_workers=True)

    # AMP scaler only needed on CUDA
    scaler = GradScaler() if use_cuda else None

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None


    print(f"Training on {len(train_idx)} samples, validating on {len(val_idx)}.")

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=use_cuda), yb.to(device, non_blocking=use_cuda)
            optimizer.zero_grad()

            if use_cuda:
                # Enable AMP on CUDA devices
                with autocast(enabled=True):
                    logits = model(xb)
                    loss = loss_fn(logits, yb)

                # Backward with scaler if available
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Fallback if scaler not available
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
        with libraries.torch.no_grad():
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