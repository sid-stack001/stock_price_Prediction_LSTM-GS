import numpy as np

def ensemble_predict(lstm_predictions, gru_predictions):
    # Weighted average of both models (can adjust weights as needed)
    return 0.5 * lstm_predictions + 0.5 * gru_predictions
