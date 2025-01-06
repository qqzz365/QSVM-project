import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Step 1: Download stock data
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    return data

# Step 2: Prepare feature and label data
def prepare_features_labels(data):
    # Moving Averages
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    
    # MACD Calculation
    short_window = 12
    long_window = 26
    signal_window = 9
    
    data['EMA_12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']  # MACD Line
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()  # Signal Line
    
    # RSI Calculation
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Drop NaN values after adding indicators
    data = data.dropna()
    
    # Features: SMA, MACD, RSI
    X = data[['SMA_5', 'SMA_10', 'MACD', 'MACD_Signal', 'RSI']].values
    y = np.where(data['Return'] > 0, 1, 0)  # Binary classification
    
    return X, y, data

# Step 3: Data preprocessing
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Build and train QSVM model
def build_qsvm(X_train, y_train, X_test, y_test):
    # Quantum feature map
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    
    # Quantum kernel using fidelity
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
    
    # QSVC model
    print("Fitting QSVM model...")
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = qsvc.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return {
        'testing_accuracy': accuracy,
        'predictions': y_pred,
        'model': qsvc
    }

# Step 5: Visualization
def plot_actual_vs_predicted(data, y_pred, test_indices):
    plt.figure(figsize=(12, 6))
    test_dates = data.index[test_indices]
    plt.plot(data.index[test_indices], data.loc[test_dates, 'Close'], label='Actual Price', color='black', alpha=0.6)
    
    # Plot predicted movements
    up_pred = test_dates[y_pred == 1]
    down_pred = test_dates[y_pred == 0]
    
    plt.scatter(up_pred, data.loc[up_pred, 'Close'], 
                color='green', marker='^', label='Predicted Up', alpha=0.7)
    plt.scatter(down_pred, data.loc[down_pred, 'Close'], 
                color='red', marker='v', label='Predicted Down', alpha=0.7)
    
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price (TWD)')
    plt.title('Stock Price with QSVM Predictions')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Step 6: Main Execution
def main():
    try:
        # Step 1: Download data
        print("Downloading stock data...")
        ticker = "0050.TW"
        start_date = "2020-12-17"
        end_date = "2024-12-17"
        data = download_stock_data(ticker, start_date, end_date)
        
        # Step 2: Prepare features and labels
        print("Preparing features and labels...")
        X, y, full_data = prepare_features_labels(data)
        
        # Step 3: Preprocess data
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        
        # Step 4: Train QSVM model
        print("Training QSVM model...")
        result = build_qsvm(X_train, y_train, X_test, y_test)
        print(f"Testing Accuracy: {result['testing_accuracy']:.2%}")
        
        # Step 5: Visualization
        print("Generating visualization...")
        test_indices = np.arange(len(full_data) - len(y_test), len(full_data))
        plot_actual_vs_predicted(full_data, result['predictions'], test_indices)
        
        # Print classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, result['predictions']))
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
