import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path, split_ratio=0.8):
    dataset = pd.read_csv(file_path)
    close_prices = dataset['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    train_size = int(len(scaled_data) * split_ratio)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - 60:]

    def create_sequences(data):
        X, Y = [], []
        for i in range(60, len(data)):
            X.append(data[i-60:i, 0])
            Y.append(data[i, 0])
        return np.array(X), np.array(Y)

    train_X, train_Y = create_sequences(train_data)
    test_X, test_Y = create_sequences(test_data)
    train_X = np.reshape(train_X, (train_X.shape[0], train_X.shape[1], 1))
    test_X = np.reshape(test_X, (test_X.shape[0], test_X.shape[1], 1))

    return train_X, train_Y, test_X, test_Y, scaler
