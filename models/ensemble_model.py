from models.lstm_model import train_and_predict_lstm
from models.gru_model import train_and_predict_gru
from networks.ensemble_network import ensemble_predict
from utils.data_preprocessing import load_data

def train_and_predict_ensemble(train_X, train_Y, test_X, scaler, epochs=20, batch_size=64):
    lstm_predictions = train_and_predict_lstm(train_X, train_Y, test_X, scaler, epochs, batch_size)
    gru_predictions = train_and_predict_gru(train_X, train_Y, test_X, scaler, epochs, batch_size)
    ensemble_predictions = ensemble_predict(lstm_predictions, gru_predictions)
    return ensemble_predictions
