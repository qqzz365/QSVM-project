# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Download stock data
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    return data

# Step 2: Prepare feature and label data
def prepare_features_labels(data):
    # Features: Moving averages and technical indicators
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()
    data = data.dropna()

    # Using more features for classical SVM
    X = data[['SMA_5', 'SMA_10']].values
    y = np.where(data['Return'] > 0, 1, 0)  # Binary classification
    return X, y, data

# Calculate RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate MACD
def calculate_macd(prices, fast=12, slow=26):
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    return exp1 - exp2

# Step 3: Data preprocessing
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Build and train SVM model
def build_svm(X_train, y_train, X_test, y_test):
    # Create SVM instance
    svm = SVC(kernel='rbf', C=1.0, gamma='scale')

    # Fit the model
    print("Fitting SVM model...")
    svm.fit(X_train, y_train)

    # Make predictions
    print("Making predictions...")
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        'testing_accuracy': accuracy,
        'predictions': y_pred,
        'model': svm
    }

# Step 5: Visualization
def plot_actual_vs_predicted(data, y_pred, test_indices, y_test):
    plt.figure(figsize=(12, 6))
    test_dates = data.index[test_indices]

    # Plot actual price
    plt.plot(test_dates, data.loc[test_dates, 'Close'], 
             label='Actual Price', color='black', alpha=0.8, linewidth=2)

    # Create masks for correct and incorrect predictions
    correct_mask = y_pred == y_test
    incorrect_mask = y_pred != y_test

    # Plot colored regions for correct/incorrect predictions
    for i in range(len(test_dates)-1):
        if correct_mask[i]:
            plt.axvspan(test_dates[i], test_dates[i+1], 
                       alpha=0.2, color='green', label='Correct' if i == 0 else "")
        else:
            plt.axvspan(test_dates[i], test_dates[i+1], 
                       alpha=0.2, color='red', label='Incorrect' if i == 0 else "")

    # Plot predicted movements with triangles
    up_pred = test_dates[y_pred == 1]
    down_pred = test_dates[y_pred == 0]

    plt.scatter(up_pred, data.loc[up_pred, 'Close'], 
               color='green', marker='^', label='Predicted Up', alpha=0.7, s=100)
    plt.scatter(down_pred, data.loc[down_pred, 'Close'], 
               color='red', marker='v', label='Predicted Down', alpha=0.7, s=100)

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price (TWD)')
    plt.title('Stock Price with Prediction Accuracy')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Step 6: Main Execution
def main():
    try:
        # Step 1: Download data
        print("Downloading stock data...")
        ticker = "0050.TW"  # 0050
        start_date = "2020-01-01"
        end_date = "2024-12-17"
        data = download_stock_data(ticker, start_date, end_date)

        # Step 2: Prepare features and labels
        print("Preparing features and labels...")
        X, y, full_data = prepare_features_labels(data)

        # Step 3: Preprocess data
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data(X, y)

        # Step 4: Train SVM model
        print("Training SVM model...")
        result = build_svm(X_train, y_train, X_test, y_test)
        print(f"Testing Accuracy: {result['testing_accuracy']:.2%}")

        # Step 5: Visualization
        print("Generating visualization...")
        test_indices = np.arange(len(full_data)-len(y_test), len(full_data))
        plot_actual_vs_predicted(full_data, result['predictions'], test_indices, y_test)

        # Print classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, result['predictions']))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
