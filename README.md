# stock_price_Prediction-GS

This project predicts stock prices using multiple deep learning models such as LSTM, GRU, and a hybrid ensemble model that integrates predictions from both models.

## Project Structure

- **data/**: Contains the historical stock price data.
- **models/**: Holds the code for each model (LSTM, GRU, and ensemble).
- **networks/**: Defines the architecture for each network.
- **utils/**: Utility functions for data preprocessing, evaluation, and visualization.
- **results/**: Stores the prediction results.
- **notebooks/**: Jupyter notebook for visualization and data analysis.
- **test/**: Scripts to test model predictions.
- **streamlit_app.py**: A Streamlit app for interactive model usage.

## Features

- **LSTM Model**: Captures long-term dependencies in time-series data.
- **GRU Model**: Similar to LSTM but with fewer parameters, making it faster for some datasets.
- **Ensemble Model**: Combines predictions from LSTM and GRU for potentially better performance.
- **Streamlit Web App**: Provides an interactive interface for visualizing actual vs. predicted stock prices.
- **Jupyter Notebook**: Includes comprehensive visualizations and step-by-step analysis.
- **Unit Testing**: Ensures the accuracy of model predictions via Mean Squared Error (MSE).

## Project Structure

```bash
stock_price_Prediction-GS/
│
├── data/
│   └── historical_data.csv             # Historical stock price data
├── models/
│   ├── lstm_model.py                   # LSTM model training and prediction
│   ├── gru_model.py                    # GRU model training and prediction
│   └── ensemble_model.py               # Ensemble model that integrates LSTM and GRU predictions
├── networks/
│   ├── lstm_network.py                 # LSTM network architecture
│   ├── gru_network.py                  # GRU network architecture
│   └── ensemble_network.py             # Ensemble model combination logic
├── utils/
│   ├── data_preprocessing.py           # Data loading and preprocessing logic
│   ├── evaluation.py                   # Evaluation metrics (e.g., MSE)
│   └── visualization.py                # Functions for plotting actual vs. predicted prices
├── results/
│   └── predictions.csv                 # Stores prediction results (if needed)
├── notebooks/
│   └── stock_prediction_analysis.ipynb # Jupyter notebook for model analysis and visualization
├── test/
│   └── test_predictions.py             # Unit tests for evaluating model predictions
├── streamlit_app.py                    # Streamlit app for interactive stock prediction
├── requirements.txt                    # Python dependencies
└── README.md                           # Project description and instructions

```

To install the required packages
   ```bash
   pip install -r requirements.txt
   ```
