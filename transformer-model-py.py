import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

class TransformerModel:
    def __init__(self, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0.2, mlp_dropout=0.4):
        self.input_shape = input_shape
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout = dropout
        self.mlp_dropout = mlp_dropout
        self.model = self.build_model()

    def transformer_encoder(self, inputs):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )(x, x)
        x = layers.Dropout(self.dropout)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build_model(self):
        inputs = keras.Input(shape=self.input_shape)
        x = inputs
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x)

        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)
        outputs = layers.Dense(1)(x)
        return keras.Model(inputs, outputs)

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                f"models/transformer_model_{int(time.time())}.h5",
                save_best_only=True
            )
        ]

        return self.model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

def main():
    # Load and preprocess data
    data = pd.read_csv("data/processed/nifty50_processed.csv", index_col="Date", parse_dates=True)
    features = data[['Open', 'High', 'Low', 'Volume', 'Close']].values
    
    # Prepare sequences
    sequence_length = 60
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i+sequence_length])
        y.append(features[i+sequence_length, 4])  # 4 is the index of 'Close' price
    X, y = np.array(X), np.array(y)

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Initialize and train model
    model = TransformerModel(
        input_shape=(sequence_length, 5),
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128, 64],
        dropout=0.2,
        mlp_dropout=0.4,
    )

    model.model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mean_absolute_error"]
    )

    history = model.train(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # Evaluate model
    mse, mae = model.evaluate(X_test, y_test)
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")

    # Make predictions
    predictions = model.predict(X_test)

    # Save predictions
    pred_df = pd.DataFrame(predictions, index=data.index[split+sequence_length:], columns=['Predicted_Close'])
    pred_df.to_csv('data/processed/transformer_predictions.csv')

    print("Transformer model training and evaluation complete.")

if __name__ == "__main__":
    main()
