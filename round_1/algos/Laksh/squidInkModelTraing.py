import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ================================
# Step 1: Data Loading and Preprocessing
# ================================
price_files = [
    "../data/prices_round_1_day_-1.csv",
    "../data/prices_round_1_day_-2.csv",
    "../data/prices_round_1_day_0.csv"
]


df_list = []
for file in price_files:
    try:
        # Read CSV using semicolon as delimiter.
        df = pd.read_csv(file, delimiter=';')
    except Exception as e:
        print(f"Error reading {file}: {e}")
        continue
    # Filter to only include SQUID_INK rows.
    df = df[df['product'] == "SQUID_INK"]
    df_list.append(df)

if not df_list:
    raise ValueError("No data loaded; please check your CSV file paths and contents.")

# Combine all DataFrames and sort by timestamp.
data = pd.concat(df_list, ignore_index=True)
data = data.sort_values(by="timestamp")
data['mid_price'] = pd.to_numeric(data['mid_price'], errors='coerce')
data = data.dropna(subset=['mid_price'])
print(f"Total SQUID_INK mid_price data points: {len(data)}")

# ================================
# Step 2: Construct Lagged Dataset
# ================================

# Use a lag window of 6: each sample uses 6 previous mid_prices as features and the current mid_price as target.
lag = 6
mid_prices = data['mid_price'].values
if len(mid_prices) < lag + 1:
    raise ValueError("Not enough data points for the lag window.")

X_list, y_list = [], []
for i in range(lag, len(mid_prices)):
    X_list.append(mid_prices[i-lag:i])  # shape: (6,)
    y_list.append(mid_prices[i])        # scalar target

X = np.array(X_list)  # Shape: (num_samples, 6)
y = np.array(y_list)  # Shape: (num_samples,)

print("Shape of feature matrix X:", X.shape)
print("Shape of target vector y:", y.shape)

# ================================
# Step 3: Normalize the Data
# ================================
# Normalize features and targets for stable training.
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_norm = (y - y_mean) / y_std

print("X_mean:", X_mean)
print("X_std:", X_std)
print("y_mean:", y_mean)
print("y_std:", y_std)

# Convert data to PyTorch tensors.
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y_norm, dtype=torch.float32).view(-1, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move data tensors to the GPU
X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)
# ================================
# Step 4: Define the Neural Network Model
# ================================
# Architecture:
#  - Input layer: 6 neurons (lag features)
#  - Hidden Layer 1: 32 neurons, ReLU activation
#  - Hidden Layer 2: 16 neurons, ReLU activation
#  - Output Layer: 1 neuron, linear activation

class PricePredictor(nn.Module):
    def __init__(self, input_dim=6, hidden1=32, hidden2=16, output_dim=1):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = PricePredictor().to(device)
print(model)

# ================================
# Step 5: Set Up Training Parameters and Train the Model
# ================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 2000

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss (normalized): {loss.item():.4f}")

# ================================
# Step 6: Evaluation
# ================================
model.eval()
with torch.no_grad():
    y_pred_norm = model(X_tensor)

# Move predictions back to CPU for further processing
y_pred_norm = y_pred_norm.cpu()
y_tensor = y_tensor.cpu()

loss_norm = criterion(y_pred_norm, y_tensor).item()
rmse_norm = np.sqrt(loss_norm)
    
# Convert predictions back to original scale.
y_pred_norm_np = y_pred_norm.numpy()
y_pred_orig = y_pred_norm_np * y_std + y_mean
rmse_orig = np.sqrt(np.mean((y - y_pred_orig.flatten()) ** 2))

print("\nFinal Loss (Normalized):", loss_norm)
print("RMSE on Normalized Data:", rmse_norm)
print("RMSE on Original Scale:", rmse_orig)

# ================================
# Step 7: Output Final Weights, Biases, and Normalization Parameters
# ================================
print("\nModel Architecture and Parameters:")
print(model)

print("\n--- Model Weights and Biases ---")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: shape {param.data.shape}")
        print(param.data)

print("\nNormalization Parameters:")
print("X_mean:", X_mean)
print("X_std:", X_std)
print("y_mean:", y_mean)
print("y_std:", y_std)
