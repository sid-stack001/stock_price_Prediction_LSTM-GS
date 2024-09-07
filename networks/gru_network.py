from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout

def create_gru(input_shape):
    model = Sequential()
    model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(GRU(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    return model
