# This file will act as the main entry point of the project. It
# will orchestrate the entire workflow by calling functions from
# the other modules: preprocessing data, training the model, and
# performing quantum optimization.

import torch
import matplotlib.pyplot as plt
from data_processing import load_austin_data, fetch_google_traffic_data, fetch_tomtom_traffic_data, merge_datasets, handle_missing_values, create_time_series, normalize_data
from model_training import LSTMModel, train_model, evaluate_model, LSTMWrapper
from hyperparameter_tuning import hyperparameter_tuning, param_grid
from quantum_optimization import grover_algorithm_with_sv
from qiskit.visualization import plot_histogram
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

# Build the LSTM model with different hyperparameters
n_timesteps = X_train_scaled.shape[1]
n_features = X_train_scaled.shape[2]

# You can modify the number of LSTM units here
model = LSTMModel(n_timesteps=n_timesteps, n_features=n_features)

# Tune the learning rate and batch size
model = train_model(
    model,
    X_train_scaled,
    y_train_scaled,
    X_test_scaled,
    y_test_scaled,
    epochs=50,      # Experiment with epochs
    batch_size=32,  # Experiment with batch size (e.g., 32, 64, 128)
    patience=5      # Keep early stopping for overfitting prevention
)

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

# Perform hyperparameter tuning
best_params = hyperparameter_tuning(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, param_grid)

# Now use the best hyperparameters to train the final model
print(f'Best Hyperparameters: {best_params}')

# Use the best hyperparameters to train your model
best_model = LSTMWrapper(
    n_timesteps=X_train_scaled.shape[1], 
    n_features=X_train_scaled.shape[2], 
    lstm_units=best_params['lstm_units'], 
    learning_rate=best_params['learning_rate']
    )

best_model.train(
    X_train_scaled, 
    y_train_scaled, 
    batch_size=best_params['batch_size'], 
    epochs=best_params['epochs']
    )

# Evaluate the best model on the test data
test_loss, mae, r2 = evaluate_model(best_model.model, X_test_scaled, y_test_scaled, scaler_y=scaler_y)
print(f"Best Model Test Loss (MSE): {test_loss}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Visualize the results (actual vs predicted values)
y_pred_scaled = best_model.model(torch.tensor(X_test_scaled, dtype=torch.float32)).detach().numpy()
y_pred_original = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Call the visualization function
plot_predictions(y_test_original, y_pred_original)

# Define the number of qubits and marked states
num_qubits = 3
marked_states = ["011", "100"] # These are binary-encoded representations of hyperparameter sets
M = 5 # Number of computational basis states

# Choose the backend
backend_choice = input("Choose backend: 'local', 'ibmq', or 'aws': ").strip().lower()

# Run Grover's algorithm with Shukla-Vedula superposition
result = grover_algorithm_with_sv(num_qubits, marked_states, M, backend_choice)

# Visualize the results
plot_histogram(result.get_counts())

