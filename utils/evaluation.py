from sklearn.metrics import mean_squared_error

def evaluate_predictions(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return mse
