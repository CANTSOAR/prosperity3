import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Step 1: Data Loading and Preprocessing
# -----------------------------
price_files = [
    "../../data/prices_round_1_day_-1.csv",
    "../../data/prices_round_1_day_-2.csv",
    "../../data/prices_round_1_day_0.csv"
]
df_list = []
for file in price_files:
    df = pd.read_csv(file, delimiter=';')
    # Filter for SQUID_INK data.
    df = df[df['product'] == "SQUID_INK"]
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)
data = data.sort_values(by="timestamp")
data['mid_price'] = pd.to_numeric(data['mid_price'], errors='coerce')
data = data.dropna(subset=['mid_price'])
print("Total SQUID_INK data points:", len(data))

# -----------------------------
# Step 2: Construct a Sequence Dataset
# -----------------------------
# Choose a sequence length (e.g., 6).
seq_length = 20
prices = data['mid_price'].values

# Create sequences: for each index, X = prices[i-seq_length:i] and target y = prices[i]
X, y = [], []
for i in range(seq_length, len(prices)):
    X.append(prices[i - seq_length:i])
    y.append(prices[i])
X = np.array(X)  # Shape: (n_samples, seq_length)
y = np.array(y)  # Shape: (n_samples,)

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Normalize the data for better training stability.
global_mean = X.mean()
global_std  = X.std()
X_norm = (X - global_mean) / global_std
y_norm = (y - global_mean) / global_std

# -----------------------------
# Step 3: Create a PyTorch Dataset and DataLoader
# -----------------------------
class PriceSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

dataset = PriceSequenceDataset(X_norm, y_norm)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# -----------------------------
# Step 4: Define the LSTM Model
# -----------------------------
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x: (batch_size, seq_length). We need shape (batch_size, seq_length, input_size)
        x = x.unsqueeze(2)  # Now: (batch_size, seq_length, 1)
        # Initialize hidden and cell states on the device.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # Forward propagate LSTM.
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_size)
        # Use the last time step's output.
        out = out[:, -1, :]  # shape: (batch_size, hidden_size)
        out = self.fc(out)   # shape: (batch_size, output_size)
        return out

model = LSTMPricePredictor(input_size=1, hidden_size=32, num_layers=1, output_size=1).to(device)
print(model)

# -----------------------------
# Step 5: Train the Model on GPU
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 500

for epoch in range(epochs):
    model.train()
    epoch_losses = []
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    
    if (epoch+1) % 20 == 0:
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs}, Loss (normalized): {avg_loss:.4f}")

# -----------------------------
# Step 6: Evaluate the Model on GPU
# -----------------------------
model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X_norm, dtype=torch.float32).to(device)
    y_pred_norm = model(X_tensor)
y_pred_norm = y_pred_norm.squeeze().cpu().numpy()
mse_norm = mean_squared_error(y_norm, y_pred_norm)
rmse_norm = np.sqrt(mse_norm)

# Convert predictions back to original scale.
y_pred_orig = y_pred_norm * global_std + global_mean
rmse_orig = np.sqrt(mean_squared_error(y, y_pred_orig))
print("\nFinal RMSE on Normalized Data:", rmse_norm)
print("Final RMSE on Original Scale:", rmse_orig)

# -----------------------------
# Step 7: Output Model Parameters and Normalization Values
# -----------------------------
print("\nModel Architecture and Parameters:")
print(model)
print("\nNormalization Parameters:")
print("Global mean:", global_mean)
print("Global std:", global_std)

# Optionally, save the model for later extraction.
torch.save(model.state_dict(), "lstm_model.pth")
