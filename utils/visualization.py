import matplotlib.pyplot as plt

def plot_predictions(actual_prices, predicted_prices, title='Stock Price Prediction'):
    plt.figure(figsize=(10, 6))
    plt.plot(actual_prices, color='blue', label='Actual Prices')
    plt.plot(predicted_prices, color='red', label='Predicted Prices')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
