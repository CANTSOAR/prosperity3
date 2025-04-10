import torch
import torch.nn as nn
torch.set_printoptions(threshold=float('inf'))
# Define the LSTM model architecture.
# Make sure this matches the architecture used during training.
class LSTMPricePredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMPricePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x should have shape: (batch_size, seq_length)
        x = x.unsqueeze(2)  # Reshape to (batch_size, seq_length, input_size)
        # Initialize hidden and cell states with zeros.
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        # Use output from the last time step.
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Create an instance of the model.
model = LSTMPricePredictor(input_size=1, hidden_size=32, num_layers=1, output_size=1)

# Load the saved model parameters.
model_path = "lstm_model_200_epochs.pth"  # change if your file is named differently.
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Print out all parameters.
print("Model parameters:")
for name, param in model.named_parameters():
    print(f"{name}:")
    print(param.data)
