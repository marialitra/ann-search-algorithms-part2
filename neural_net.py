import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

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


# Simple Training

# def train_model(args, dataset_dimension, X, y):
#     # Select computation device (GPU if available) 
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Initialize model, optimizer, and loss function 
#     model = MLPClassifier(d_in=dataset_dimension, n_out=args.m, neurons=args.nodes, n_layers=args.layers).to(device)

#     # Initialize the optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
#     loss_fn = nn.CrossEntropyLoss()

#     dataset = torch.utils.data.TensorDataset(torch.from_numpy(X.copy()).float(), torch.from_numpy(y.copy()).long())
#     loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


#     for epoch in range(args.epochs):
#         model.train()
#         total_loss = 0.0

#         # Main training loop 
#         for xd, yd in loader: 
#             # Move data to device 
#             xd, yd = xd.to(device), yd.to(device)

#             # 1. Reset previous gradients 
#             optimizer.zero_grad()

#             # 2. Forward pass
#             logits = model(xd)

#             # 3. Compute loss
#             loss = loss_fn(logits, yd)
            
#             # 4. Backward pass (gradients)
#             loss.backward()

#             # 5. Update parameters
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")

#     model = model.to(torch.device("cpu"))
#     return model



# Training with splitting training data into train and val data

# def train_model(args, dataset_dimension, X, y):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = MLPClassifier(d_in=dataset_dimension, n_out=args.m,
#                           neurons=args.nodes, n_layers=args.layers).to(device)

#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     loss_fn = nn.CrossEntropyLoss()

#     # Split data into train / val
#     X_train, X_val, y_train, y_val = train_test_split(
#         X, y, test_size=0.1, random_state=42, stratify=y
#     )

#     train_ds = torch.utils.data.TensorDataset(
#         torch.from_numpy(X_train.copy()).float(), torch.from_numpy(y_train.copy()).long())
#     val_ds = torch.utils.data.TensorDataset(
#         torch.from_numpy(X_val.copy()).float(), torch.from_numpy(y_val.copy()).long())

#     train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
#     val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)

#     best_val_loss = float("inf")
#     patience_counter = 0
#     patience = 3  # stop if val loss doesn't improve for 3 epochs

#     for epoch in range(args.epochs):
#         model.train()
#         total_loss = 0.0
#         for xd, yd in train_loader:
#             xd, yd = xd.to(device), yd.to(device)
#             optimizer.zero_grad()
#             logits = model(xd)
#             loss = loss_fn(logits, yd)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         avg_train_loss = total_loss / len(train_loader)

#         # ---- validation phase ----
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for xv, yv in val_loader:
#                 xv, yv = xv.to(device), yv.to(device)
#                 logits = model(xv)
#                 val_loss += loss_fn(logits, yv).item()
#         avg_val_loss = val_loss / len(val_loader)

#         print(f"Epoch {epoch+1}/{args.epochs} "
#               f"- Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

#         # early stopping check
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print("Early stopping triggered.")
#                 break

#     model = model.to(torch.device("cpu"))
#     return model


# K folds training (best way till now)

# from sklearn.model_selection import KFold
# import torch
# import torch.nn as nn
# import numpy as np

# def train_model(args, dataset_dimension, X, y):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     kfold = KFold(n_splits=getattr(args, "kfolds", 4), shuffle=True, random_state=42)

#     fold_results = []

#     for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
#         print("-" * 60)
#         print(f"Fold {fold+1}/{kfold.get_n_splits()}")

#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         model = MLPClassifier(d_in=dataset_dimension, n_out=args.m,
#                               neurons=args.nodes, n_layers=args.layers).to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#         loss_fn = nn.CrossEntropyLoss()

#         train_ds = torch.utils.data.TensorDataset(
#             torch.from_numpy(X_train.copy()).float(), torch.from_numpy(y_train.copy()).long())
#         val_ds = torch.utils.data.TensorDataset(
#             torch.from_numpy(X_val.copy()).float(), torch.from_numpy(y_val.copy()).long())

#         train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
#         val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)

#         best_val_loss = float("inf")
#         patience_counter = 0
#         patience = 3

#         for epoch in range(args.epochs):
#             model.train()
#             total_loss = 0.0
#             for xd, yd in train_loader:
#                 xd, yd = xd.to(device), yd.to(device)
#                 optimizer.zero_grad()
#                 logits = model(xd)
#                 loss = loss_fn(logits, yd)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()

#             avg_train_loss = total_loss / len(train_loader)

#             # Validation loss
#             model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for xv, yv in val_loader:
#                     xv, yv = xv.to(device), yv.to(device)
#                     val_loss += loss_fn(model(xv), yv).item()
#             avg_val_loss = val_loss / len(val_loader)

#             print(f"Epoch {epoch+1}/{args.epochs} - "
#                   f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 patience_counter = 0
#                 best_model_state = model.state_dict()
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     print("Early stopping triggered.")
#                     break

#         fold_results.append(best_val_loss)
#         print(f"Fold {fold+1} best val loss: {best_val_loss:.4f}")

#     print("-" * 60)
#     print(f"Average validation loss across folds: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

#     # retrain on all data with best hyperparameters (optional)
#     print("Retraining on full dataset with best fold settings...")
#     model = MLPClassifier(d_in=dataset_dimension, n_out=args.m,
#                           neurons=args.nodes, n_layers=args.layers).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#     loss_fn = nn.CrossEntropyLoss()

#     full_ds = torch.utils.data.TensorDataset(
#         torch.from_numpy(X.copy()).float(), torch.from_numpy(y.copy()).long())
#     full_loader = torch.utils.data.DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)

#     for epoch in range(args.epochs):
#         model.train()
#         total_loss = 0.0
#         for xd, yd in full_loader:
#             xd, yd = xd.to(device), yd.to(device)
#             optimizer.zero_grad()
#             logits = model(xd)
#             loss = loss_fn(logits, yd)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"[Final Model] Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(full_loader):.4f}")

#     model = model.to("cpu")
#     return model


#kfolds with regularization and dropout

# import torch
# import torch.nn as nn
# import numpy as np
# from sklearn.model_selection import KFold

# # Define a simple feedforward neural network (MLP) 
# class MLPClassifier(nn.Module): 
#     # Added dropout_rate argument for regularization
#     def __init__(self, d_in, n_out, neurons, n_layers, dropout_rate=0.25): 
#         super().__init__() 

#         layers = []
#         # Input layer
#         layers.append(nn.Linear(d_in, neurons))
#         layers.append(nn.ReLU())
#         layers.append(nn.Dropout(dropout_rate)) # Dropout after activation
        
#         # Hidden layers
#         for _ in range(n_layers - 2):
#             layers.append(nn.Linear(neurons, neurons))
#             layers.append(nn.ReLU())
#             layers.append(nn.Dropout(dropout_rate)) # Dropout after activation

#         # Output layer
#         layers.append(nn.Linear(neurons, n_out))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x): 
#         # Forward pass through the network 
#         return self.net(x)


# def train_model(args, dataset_dimension, X, y):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # --- Configuration for Regularization (Check args for customization) ---
#     dropout_rate = getattr(args, "dropout_rate", 0.25)
#     weight_decay = getattr(args, "weight_decay", 1e-5) # L2 regularization

#     kfold = KFold(n_splits=getattr(args, "kfolds", 4), shuffle=True, random_state=42)

#     fold_results = []

#     for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
#         print("-" * 60)
#         print(f"Fold {fold+1}/{kfold.get_n_splits()}")

#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         # Pass dropout_rate to the classifier
#         model = MLPClassifier(d_in=dataset_dimension, n_out=args.m,
#                               neurons=args.nodes, n_layers=args.layers, 
#                               dropout_rate=dropout_rate).to(device)
                              
#         # Add weight_decay to the optimizer for L2 regularization
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
#         loss_fn = nn.CrossEntropyLoss()

#         train_ds = torch.utils.data.TensorDataset(
#             torch.from_numpy(X_train.copy()).float(), torch.from_numpy(y_train.copy()).long())
#         val_ds = torch.utils.data.TensorDataset(
#             torch.from_numpy(X_val.copy()).float(), torch.from_numpy(y_val.copy()).long())

#         train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
#         val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size)

#         best_val_loss = float("inf")
#         patience_counter = 0
#         patience = 3
#         best_model_state = None

#         for epoch in range(args.epochs):
#             model.train()
#             total_loss = 0.0
#             for xd, yd in train_loader:
#                 xd, yd = xd.to(device), yd.to(device)
#                 optimizer.zero_grad()
#                 logits = model(xd)
#                 loss = loss_fn(logits, yd)
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()

#             avg_train_loss = total_loss / len(train_loader)

#             # Validation loss
#             model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for xv, yv in val_loader:
#                     xv, yv = xv.to(device), yv.to(device)
#                     val_loss += loss_fn(model(xv), yv).item()
#             avg_val_loss = val_loss / len(val_loader)

#             print(f"Epoch {epoch+1}/{args.epochs} - "
#                   f"Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

#             if avg_val_loss < best_val_loss:
#                 best_val_loss = avg_val_loss
#                 patience_counter = 0
#                 # Save the state_dict for the best model in this fold
#                 best_model_state = model.state_dict() 
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     print("Early stopping triggered.")
#                     break

#         fold_results.append(best_val_loss)
#         print(f"Fold {fold+1} best val loss: {best_val_loss:.4f}")

#     print("-" * 60)
#     print(f"Average validation loss across folds: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")

#     # Retrain on full dataset
#     print("Retraining on full dataset with final configuration...")
#     model = MLPClassifier(d_in=dataset_dimension, n_out=args.m,
#                           neurons=args.nodes, n_layers=args.layers, 
#                           dropout_rate=dropout_rate).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
#     loss_fn = nn.CrossEntropyLoss()

#     full_ds = torch.utils.data.TensorDataset(
#         torch.from_numpy(X.copy()).float(), torch.from_numpy(y.copy()).long())
#     full_loader = torch.utils.data.DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)

#     for epoch in range(args.epochs):
#         model.train()
#         total_loss = 0.0
#         for xd, yd in full_loader:
#             xd, yd = xd.to(device), yd.to(device)
#             optimizer.zero_grad()
#             logits = model(xd)
#             loss = loss_fn(logits, yd)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
        
#         # Simple logging for full training, no validation
#         print(f"[Final Model] Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(full_loader):.4f}")

#     model = model.to("cpu")
#     return model


# kfolds, regularization, dropout, cnn and pooling
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold

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