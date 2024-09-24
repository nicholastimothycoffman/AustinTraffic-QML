# This file will focus on defining and training the neural
# network. It will use the preprocessed data and build the
# model architecture (LSTM), while also allowing for
# quantum enhanced hyperparameter tuning.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

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
def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=64, patience=3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert the data to PyTorch tensors and create DataLoader
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

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
        
        # Calcuate the validation loss
        val_loss, _, _ = evaluate_model(model, X_test, y_test)
        train_loss = running_loss / len(train_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth') # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping...")
                break

    # Plot training loss over time
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.legend()
    plt.show()

    return model

# Function to load the model (for later use or evaluation):
def load_model(filepath, n_timesteps, n_features):
    model = LSTMModel(n_timesteps=n_timesteps, n_features=n_features)
    model.load_state_dict(torch.load('lstm_model.pth'))
    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test, scaler_y=None):
    model.eval()
    criterion = nn.MSELoss()
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(X_test)
        test_loss = criterion(outputs, y_test.unsqueeze(1))

    # Convert outputs to numpy arrays for evaluation with sklearn
    y_pred_scaled = outputs.numpy()
    y_test_scaled = y_test.numpy()

    # Inverse-transform predictions if targets were scaled
    if scaler_y:
        y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = scaler_y.inverse_transform(y_test_scaled)
    else:
        y_pred_original = y_pred_scaled
        y_test_original = y_test_scaled

    # Calculate MAE and R²
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    print(f'Test MSE: {test_loss.item()}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'R² Score: {r2}')

    return test_loss.item(), mae, r2