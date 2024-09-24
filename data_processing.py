# This file will handle all tasks related to data loading, 
# cleaning, and feature engineering from the different 
# sources (Austin Open Data, Google Maps API, TomTom API
# INRIX).

import pandas as pd
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_austin_data(filepath):
    try:
        # Load Austin traffic data from CSV
        data = pd.read_csv(filepath)
        # Forward-fill to handle missing values
        data.fillna(method='ffill', inplace=True)
        # Drop rows with missing critical values
        data.dropna(inplace=True)
        return data
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File {filepath} is empty.")
        return None
    except Exception as e:
        print(f"An error occured: {e}")
        return None
    
def fetch_google_traffic_data(api_url, api_key):
    try:
        response = requests.get(f"{api_url}?key={api_key}")
        response.raise_for_status() # Raise an error if the response code is not 200
        data = response.json()
        # Process the data as needed
        # E.g., Convert timestamps to a uniform format
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Google data: {e}")
        return None

def fetch_tomtom_traffic_data(api_url, api_key):
    try:
        response = requests.get(f"{api_url}?key={api_key}")
        response.raise_for_status()
        data = response.json()
        # Process the data as needed
        # E.g., Convert timestamps to a uniform format
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching TomTom data: {e}")
        return None
    
def merge_datasets(austin_data, google_data, tomtom_data):
    try:
        # Ensure Google and TomTom dataframes exist and have proper timestamps and locations
        google_df = pd.DataFrame(google_data)
        tomtom_df = pd.DataFrame(tomtom_data)

        # Convert timestamps if necessary
        austin_data['timestamp'] = pd.to_datetime(austin_data['timestamp'])
        google_df['timestamp'] = pd.to_datetime(google_df['timestamp'])
        tomtom_df['timestamp'] = pd.to_datetime(tomtom_df['timestamp'])

        # Merge datasets on location and time
        combined_data = pd.merge(austin_data, google_df, on=['location', 'timestamp'], how='inner')
        combined_data = pd.merge(combined_data, tomtom_df, on=['location', 'timestamp'], how='inner')
        return combined_data
    except KeyError as e:
        print(f"Error during merge: missing column {e}")
        return None
    except Exception as e:
        print(f"An error occurred while merging datasets: {e}")
        return None
    
def handle_missing_values(data):
    # Interpolate missing values
    data.interpolate(method='linear', inplace=True)
    
    # Drop rows with remaining missing values
    data.dropna(inplace=True)
    
    return data


def create_time_series(data, n_timesteps=10):
    """
    data: DataFrame with traffic data and relevant features.
    n_timesteps: Number of past time steps to use for prediction.
    
    Returns:
    X: Time-series data (features).
    y: Labels (what you're predicting, e.g., future traffic count).
    """
    features = data.drop(columns=['timestamp', 'location']) # Use only numeric columns for the LSTM

    X, y = [], []
    for i in range(len(features) - n_timesteps):
        X.append(features.iloc[i:i + n_timesteps].values)
        y.append(features.iloc[i + n_timesteps].values[0]) # Predicting the next traffic value

    return np.array(X), np.array(y)

# Normalize data to range [0, 1]
def normalize_data(X_train, X_test, y_train=None, y_test=None):
    """
    Scales the features of X_train and X_test (and optionally y_train, y_test) to a [0, 1] range.

    Parameters:
    - X_train: Training data (numpy array)
    - X_test: Test data (numpy array)
    - y_train: (Optional) Training target data
    - y_test: (Optional) Test target data

    Returns:
    - X_train_scaled: Scaled training data
    - X_test_scaled: Scaled test data
    - y_train_scaled: Scaled training target data (if provided)
    - y_test_scaled: Scaled test target data (if provided)
    - scaler_X: Fitted scaler for X to inverse-transform if needed
    - scaler_y: Fitted scaler for y (if provided) to inverse-transform predictions
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler() if y_train is not None else None

    X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    if y_train is not None and y_test is not None:
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

    return X_train_scaled, X_test_scaled, scaler_X

