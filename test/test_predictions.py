import unittest
import numpy as np
from utils.data_preprocessing import load_data
from models.lstm_model import train_and_predict_lstm
from models.gru_model import train_and_predict_gru
from models.ensemble_model import train_and_predict_ensemble
from utils.evaluation import evaluate_predictions

class TestStockPricePrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load data for testing
        cls.train_X, cls.train_Y, cls.test_X, cls.test_Y, cls.scaler = load_data('../data/historical_data.csv')

    def test_lstm_predictions(self):
        lstm_predictions = train_and_predict_lstm(self.train_X, self.train_Y, self.test_X, self.scaler)
        actual_prices = self.scaler.inverse_transform(self.test_X.reshape(-1, 1))
        mse = evaluate_predictions(actual_prices, lstm_predictions)
        print(f"LSTM Model MSE: {mse}")
        self.assertTrue(mse < 0.05)

    def test_gru_predictions(self):
        gru_predictions = train_and_predict_gru(self.train_X, self.train_Y, self.test_X, self.scaler)
        actual_prices = self.scaler.inverse_transform(self.test_X.reshape(-1, 1))
        mse = evaluate_predictions(actual_prices, gru_predictions)
        print(f"GRU Model MSE: {mse}")
        self.assertTrue(mse < 0.05)

    def test_ensemble_predictions(self):
        ensemble_predictions = train_and_predict_ensemble(self.train_X, self.train_Y, self.test_X, self.scaler)
        actual_prices = self.scaler.inverse_transform(self.test_X.reshape(-1, 1))
        mse = evaluate_predictions(actual_prices, ensemble_predictions)
        print(f"Ensemble Model MSE: {mse}")
        self.assertTrue(mse < 0.05)

if __name__ == '__main__':
    unittest.main()
