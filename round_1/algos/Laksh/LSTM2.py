import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import copy

# -----------------------------
# STEP 1: Data Loading & Cleaning
# -----------------------------
# File paths to your price CSV files.
price_files = [
    "../../data/prices_round_1_day_-1.csv",
    "../../data/prices_round_1_day_-2.csv",
    "../../data/prices_round_1_day_0.csv"
]

price_dfs = []
for file in price_files:
    df = pd.read_csv(file, delimiter=';')
    # Filter to keep only SQUID_INK rows.
    df = df[df['product'] == "SQUID_INK"]
    # Convert essential columns to numeric.
    for col in ['day', 'timestamp', 'mid_price', 'bid_price_1', 'ask_price_1']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    price_dfs.append(df)

# Combine and sort by day and timestamp.
prices_df = pd.concat(price_dfs, ignore_index=True)
prices_df = prices_df.sort_values(by=["day", "timestamp"])
prices_df.dropna(subset=['mid_price', 'bid_price_1', 'ask_price_1'], inplace=True)
print("Prices data points after cleaning:", len(prices_df))

# -----------------------------
# STEP 2: Feature Engineering
# -----------------------------
# Compute the spread.
prices_df["spread"] = prices_df["ask_price_1"] - prices_df["bid_price_1"]

# Compute a rolling moving average and volatility over a window of 3.
window = 3
prices_df["moving_avg"] = prices_df["mid_price"].rolling(window=window).mean()
prices_df["volatility"] = prices_df["mid_price"].rolling(window=window).std()

# Drop rows where rolling calculations are NaN.
prices_df = prices_df.dropna(subset=["moving_avg", "volatility"])
prices_df = prices_df.sort_values(by=["day", "timestamp"])
print("Final feature dataframe shape:", prices_df.shape)
print(prices_df.head())

# -----------------------------
# STEP 3: Build Feature Matrix and Target
# -----------------------------
# For our model we choose a simple feature set:
#  - mid_price, spread, moving_avg, volatility
feature_cols = ["mid_price", "spread", "moving_avg", "volatility"]
features = prices_df[feature_cols].values
target = prices_df["mid_price"].values   # We'll forecast future mid_price values.

if features.shape[0] == 0:
    raise ValueError("The features array is empty. Check your data filtering and merging steps.")

# -----------------------------
# STEP 4: Create Sequences for Multi-Step Forecasting
# -----------------------------
# We set:
seq_length = 10      # Use the previous 10 timesteps as input.
forecast_horizon = 5 # Predict the next 5 mid_price values.
X_seq = []
y_seq = []
for i in range(seq_length, len(features) - forecast_horizon + 1):
    # Input: sequence of features from i-seq_length to i (not inclusive).
    X_seq.append(features[i-seq_length:i])
    # Target: next forecast_horizon mid_prices.
    y_seq.append(target[i:i+forecast_horizon])
X_seq = np.array(X_seq)  # Expected shape: (n_samples, seq_length, num_features)
y_seq = np.array(y_seq)  # Expected shape: (n_samples, forecast_horizon)
print("Shape of sequence data X_seq:", X_seq.shape)
print("Shape of targets y_seq:", y_seq.shape)

# -----------------------------
# STEP 5: Normalize the Data
# -----------------------------
# Normalize X (features) using overall mean and std computed over all timesteps and features.
X_mean = X_seq.mean(axis=(0, 1))
X_std = X_seq.std(axis=(0, 1))
X_norm = (X_seq - X_mean) / X_std

# Normalize y (target) globally.
y_mean = y_seq.mean()
y_std = y_seq.std()
y_norm = (y_seq - y_mean) / y_std

# -----------------------------
# STEP 6: Create PyTorch Dataset and DataLoader
# -----------------------------
class PriceSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

dataset = PriceSequenceDataset(X_norm, y_norm)
batch_size = 32
# Modify STEP 6: Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

# Create separate datasets and dataloaders for training and validation
train_dataset = PriceSequenceDataset(train_X, train_y)
val_dataset = PriceSequenceDataset(val_X, val_y)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# Suppose X_seq is a numpy array of shape (n_samples, seq_length, num_features)
# For example:
# X_seq.shape might be (n_samples, 10, 4) where 4 corresponds to: [mid_price, spread, moving_avg, volatility]

# Compute global means and stds over all samples and time steps:
GLOBAL_MEAN = X_seq.mean(axis=(0, 1))  # Shape will be (4,)
GLOBAL_STD = X_seq.std(axis=(0, 1))    # Shape will be (4,)

print("GLOBAL_MEAN:", GLOBAL_MEAN)
print("GLOBAL_STD:", GLOBAL_STD)

# -----------------------------
# STEP 7: Define the LSTM Model
# -----------------------------
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, output_size=forecast_horizon):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: shape (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)
        out = out[:, -1, :]  # Use the hidden state from the last time step.
        out = self.fc(out)   # Output shape: (batch_size, forecast_horizon)
        return out

# -----------------------------
# STEP 8: Set Up and Train the Model on GPU
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
input_size = X_norm.shape[2]  # Number of features (4 here).
model = LSTMPricePredictor(input_size=input_size, hidden_size=32, num_layers=1, output_size=forecast_horizon).to(device)
print(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100
patience = 10  # Number of epochs to wait for improvement
min_delta = 0.001  # Minimum change to qualify as an improvement
best_loss = float('inf')
counter = 0
best_model_weights = None

# Training loop with early stopping
for epoch in range(epochs):
    # Training phase
    model.train()
    train_losses = []
    for batch_X, batch_y in train_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    # Validation phase
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_X, batch_y in val_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            val_loss = criterion(outputs, batch_y)
            val_losses.append(val_loss.item())
    
    # Calculate average losses
    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)
    
    # Print progress
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Early stopping check
    if avg_val_loss < best_loss - min_delta:
        best_loss = avg_val_loss
        counter = 0
        # Save best model weights
        best_model_weights = copy.deepcopy(model.state_dict())
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_loss:.4f}")
            break

# -----------------------------
# STEP 9: Evaluate the Model
# -----------------------------
model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    y_pred_norm = model(X_tensor)
y_pred_norm = y_pred_norm.cpu().numpy()  # Shape: (n_samples, forecast_horizon)
mse_norm = mean_squared_error(y_norm.flatten(), y_pred_norm.flatten())
rmse_norm = np.sqrt(mse_norm)

# Convert predictions back to original scale.
y_pred_orig = y_pred_norm * y_std + y_mean
rmse_orig = np.sqrt(mean_squared_error(y_seq.flatten(), y_pred_orig.flatten()))
print("\nFinal RMSE on Normalized Data:", rmse_norm)
print("Final RMSE on Original Scale:", rmse_orig)

# -----------------------------
# STEP 10: Output Model & Normalization Parameters
# -----------------------------
print("\nModel Architecture:")
print(model)
print("\nNormalization Parameters (for X):")
print("X_mean:", X_mean)
print("X_std:", X_std)
print("\nNormalization Parameters (for y):")
print("y_mean:", y_mean)
print("y_std:", y_std)

weights_file = "lstm_model_weights.txt"
with open(weights_file, "w") as f:
    for name, param in model.named_parameters():
        f.write(f"Layer: {name}\n")
        f.write(f"Values: {param.detach().cpu().numpy().tolist()}\n\n")
print(f"Model weights saved to {weights_file}")

# Optionally, plot predictions vs. actual values (using the first 100 predictions as an example).
plt.figure(figsize=(10,5))
plt.plot(y_seq.flatten()[:100], label="Actual")
plt.plot(y_pred_orig.flatten()[:100], label="Predicted", alpha=0.7)
plt.title("Actual vs. Predicted SQUID_INK Mid Price (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Mid Price")
plt.legend()
plt.show()
