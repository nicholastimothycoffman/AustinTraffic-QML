from sklearn.metrics import mean_squared_error
from model_training import LSTMWrapper  # Import the wrapper you defined earlier

# Function to evaluate hyperparameters
def hyperparameter_tuning(X_train, y_train, X_test, y_test, param_grid):
    best_params = None
    best_score = float('inf')

    # Loop over all combinations of hyperparameters
    for lstm_units in param_grid['lstm_units']:
        for learning_rate in param_grid['learning_rate']:
            for batch_size in param_grid['batch_size']:
                for epochs in param_grid['epochs']:
                    # Initialize and train the model
                    model = LSTMWrapper(n_timesteps=X_train.shape[1], n_features=X_train.shape[2], lstm_units=lstm_units, learning_rate=learning_rate)
                    model.train(X_train, y_train, batch_size=batch_size, epochs=epochs)
                    
                    # Evaluate on the test set
                    y_pred = model.evaluate(X_test, y_test).numpy()
                    y_test_np = y_test.numpy()
                    
                    # Calculate the MSE
                    mse = mean_squared_error(y_test_np, y_pred)
                    print(f"MSE: {mse}, Params: {lstm_units}, {learning_rate}, {batch_size}, {epochs}")
                    
                    # Keep track of the best-performing model
                    if mse < best_score:
                        best_score = mse
                        best_params = {
                            'lstm_units': lstm_units,
                            'learning_rate': learning_rate,
                            'batch_size': batch_size,
                            'epochs': epochs
                        }
    
    print(f'Best Hyperparameters: {best_params}, Best Score: {best_score}')
    return best_params

# Parameter grid for tuning
param_grid = {
    'lstm_units': [50, 100, 150],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [32, 64, 128],
    'epochs': [50, 100]
}

# Example usage (you can call this function from main.py):
# best_params = hyperparameter_tuning(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, param_grid)
