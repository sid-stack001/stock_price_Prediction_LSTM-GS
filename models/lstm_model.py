from networks.lstm_network import create_lstm
from utils.data_preprocessing import load_data
import numpy as np

def train_and_predict_lstm(train_X, train_Y, test_X, scaler, epochs=20, batch_size=64):
    model = create_lstm(input_shape=(train_X.shape[1], 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size)
    predicted_prices = model.predict(test_X)
    return scaler.inverse_transform(predicted_prices)
