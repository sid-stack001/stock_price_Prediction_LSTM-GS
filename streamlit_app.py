import streamlit as st
import pandas as pd
from models.lstm_model import train_and_predict_lstm
from models.gru_model import train_and_predict_gru
from models.ensemble_model import train_and_predict_ensemble
from utils.data_preprocessing import preprocess_data
from utils.visualization import plot_predictions

# Streamlit app title
st.title('Stock Price Prediction with LSTM, GRU, and Ensemble')

# Upload CSV data file
uploaded_file = st.file_uploader("Upload your stock data (CSV)", type=["csv"])

# Model selection
model_type = st.selectbox('Choose model type', ['LSTM', 'GRU', 'Ensemble'])

# Model parameters
epochs = st.slider('Select number of epochs', min_value=5, max_value=100, value=20)
batch_size = st.slider('Select batch size', min_value=16, max_value=128, value=64)

if uploaded_file:
    # Load the uploaded data
    data = pd.read_csv(uploaded_file)
    st.write("**Preview of uploaded data:**")
    st.dataframe(data.head())

    # Preprocess data
    train_X, train_Y, test_X, test_Y, scaler = preprocess_data(data)

    # Train and predict based on the selected model
    if model_type == 'LSTM':
        predictions = train_and_predict_lstm(train_X, train_Y, test_X, scaler, epochs, batch_size)
    elif model_type == 'GRU':
        predictions = train_and_predict_gru(train_X, train_Y, test_X, scaler, epochs, batch_size)
    else:
        predictions = train_and_predict_ensemble(train_X, train_Y, test_X, scaler, epochs, batch_size)

    # Plot results
    actual_prices = scaler.inverse_transform(test_X.reshape(-1, 1))
    plot_predictions(actual_prices, predictions, title=f'{model_type} Stock Price Prediction')
