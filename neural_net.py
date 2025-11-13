import torch, torch.nn as nn

# Define a simple feedforward neural network (MLP) 
class MLPClassifier(nn.Module): 
    def __init__(self, d_in, n_out): 
        super().__init__() 
        # Sequential block: Linear −> ReLU −> Linear −> ReLU −> Linear 
        self.net = nn.Sequential( 
            nn.Linear(d_in, 512), # Input layer 
            nn.ReLU(),
            nn.Linear(512, 512), # Hidden layer 
            nn.ReLU(), 
            nn.Linear(512, n_out) # Output logits 
            )

    def forward(self, x): 
        # Forward pass through the network 
        return self.net(x)

def train_model(loader, m):
    # Select computation device (GPU if available) 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize model, optimizer, and loss function 
        model = MLPClassifier(d_in=128, n_out=m).to(device) 
        opt = torch.optim.Adam(model.parameters(), lr=1e−3) 
        loss_fn = nn.CrossEntropyLoss()
    # Main training loop 
    for x, y in loader: 
        x, y = x.to(device), y.to(device) # Move data to device 
        opt.zero_grad()
    # 1. Reset previous gradients 
        logits = model(x)# 2. Forward pass
        loss = loss_fn(logits, y)  # 3. Compute loss
        loss.backward()# 4. Backward pass (gradients)
        opt.step() # 5. Update parameters
    return model
