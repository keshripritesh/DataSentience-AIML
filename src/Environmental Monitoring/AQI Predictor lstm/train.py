# train.py

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from preprocess import load_and_preprocess
import os
from keras.losses import MeanSquaredError
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    

    model.compile(optimizer='adam', loss=MeanSquaredError())

    return model

if __name__ == "__main__":
    X, y = load_and_preprocess("data/final_dataset.csv", past_days=7)
    
    model = build_model((X.shape[1], X.shape[2]))
    es = EarlyStopping(patience=10, restore_best_weights=True)

    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es])

    os.makedirs("model", exist_ok=True)
    model.save("model/lstm_aqi_model.h5")
    print("✅ Model saved to model/lstm_aqi_model.h5")
