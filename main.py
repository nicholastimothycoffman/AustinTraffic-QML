# This file will act as the main entry point of the project. It
# will orchestrate the entire workflow by calling functions from
# the other modules: preprocessing data, training the model, and
# performing quantum optimization.

import torch
import matplotlib.pyplot as plt
from data_processing import load_austin_data, fetch_google_traffic_data, fetch_tomtom_traffic_data, merge_datasets, handle_missing_values, create_time_series, normalize_data
from model_training import LSTMModel, train_model, evaluate_model
from quantum_optimization import grover_search
from sklearn.model_selection import train_test_split

# Function to plot predictions
def plot_predictions(y_test_original, y_pred_original):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original, label='Actual Traffic')
    plt.plot(y_pred_original, label='Predicted Traffic', linestyle='--')
    plt.xlabel('Time Steps')
    plt.ylabel('Traffic Count')
    plt.title('Actual vs Predicted Traffic Data')
    plt.legend()
    plt.show()


# Load and preprocess data
austin_data = load_austin_data('austin_traffic_counts.csv')
google_data = fetch_google_traffic_data('google_api_url', 'your_google_api_key')
tomtom_data = fetch_tomtom_traffic_data('tomtom_api_url', 'your_tomtom_api_key')

combined_data = merge_datasets(austin_data, google_data, tomtom_data)

# Handle any missing values in the combined dataset
cleaned_data = handle_missing_values(combined_data)

# Create time_series data
X, y = create_time_series(cleaned_data, n_timesteps=10)

# Split data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = normalize_data(X_train, X_test, y_train, y_test)

# Build the LSTM model with early stopping
n_timesteps = X_train_scaled.shape[1]
n_features = X_train_scaled.shape[2]
model = LSTMModel(n_timesteps=n_timesteps, n_features=n_features)

# Train the model with early stopping
model = train_model(model, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, epochs=50, batch_size=64, patience=5)

# Load the best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate the model after training
test_loss = mae, r2 = evaluate_model(model, X_test_scaled, y_test_scaled, scaler_y=scaler_y)

# Print the results
print(f"Test Loss (MSE): {test_loss}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Inverse-transform predictions back to the original scale
y_pred_scaled = model(torch.tensor(X_test_scaled, dtype=torch.float32)).detach().numpy()
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Visualize the results
plot_predictions(y_test_original, y_pred_original)

# Save the trained model
torch.save(model.state_dict(), 'lstm_model.pth')

# Quantum-enhanced hyperparameter tuning
# Define oracle for Grover's search (implement your oracle logic)
oracle = ...
optimal_params = grover_search(oracle, num_qubits=4)
print(f'Optimal Hyperparameters: {optimal_params}')
