import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import os

class LSTMModel:
    def __init__(self, look_back=60, units=50, dropout=0.2, learning_rate=0.001):
        self.look_back = look_back
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_data(self, data):
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i])
            y.append(scaled_data[i])
        
        X, y = np.array(X), np.array(y)
        return X, y

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(units=self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(units=self.units, return_sequences=True),
            Dropout(self.dropout),
            LSTM(units=self.units),
            Dropout(self.dropout),
            Dense(units=1)
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        return model

    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        if self.model is None:
            self.model = self.build_model((X.shape[1], X.shape[2]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint('models/lstm_model.h5', save_best_only=True)
        
        history = self.model.fit(
            X, y, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=validation_split,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        return history

    def predict(self, X):
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(predictions)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred)**2)
        mae = np.mean(np.abs(y - y_pred))
        return {'MSE': mse, 'MAE': mae}

def main():
    # Load data
    data = pd.read_csv("data/processed/nifty50_processed.csv", index_col="Date", parse_dates=True)
    
    # Prepare features and target
    features = data[['Open', 'High', 'Low', 'Volume']].values
    target = data['Close'].values.reshape(-1, 1)
    
    # Initialize and train model
    model = LSTMModel(look_back=60, units=100, dropout=0.3, learning_rate=0.001)
    X, y = model.prepare_data(np.hstack((features, target)))
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    history = model.train(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)
    
    # Evaluate model
    evaluation = model.evaluate(X_test, y_test)
    print(f"Test MSE: {evaluation['MSE']:.4f}")
    print(f"Test MAE: {evaluation['MAE']:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Save predictions
    pred_df = pd.DataFrame(predictions, index=data.index[split+model.look_back:], columns=['Predicted_Close'])
    pred_df.to_csv('data/processed/lstm_predictions.csv')
    
    print("LSTM model training and evaluation complete.")

if __name__ == "__main__":
    main()    