# This file will focus on defining and training the neural
# network. It will use the preprocessed data and build the
# model architecture (LSTM), while also allowing for
# quantum enhanced hyperparameter tuning.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, n_timesteps, n_features):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(n_features, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Get the last output from the LSTM
        return out

# Function to train the model
def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert the data to PyTorch tensors and create DataLoader
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    train_losses = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}')

    # Plot training loss over time
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

# Function to load the model (for later use or evaluation):
def load_model(filepath, n_timesteps, n_features):
    model = LSTMModel(n_timesteps=n_timesteps, n_features=n_features)
    model.load_state_dict(torch.load('lstm_model.pth'))
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    model.eval()
    criterion = nn.MSELoss()
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X_test)
        test_loss = criterion(outputs, y_test.unsqueeze(1))
    
    print(f'Test MSE: {test_loss.item()}')
    return test_loss.item()