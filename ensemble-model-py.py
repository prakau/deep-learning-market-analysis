import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

class EnsembleModel:
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X, y):
        self.base_predictions = np.column_stack([
            model.predict(X).flatten() for model in self.base_models
        ])
        self.meta_model.fit(self.base_predictions, y)

    def predict(self, X):
        base_predictions = np.column_stack([
            model.predict(X).flatten() for model in self.base_models
        ])
        return self.meta_model.predict(base_predictions)

def load_predictions(lstm_path, transformer_path):
    lstm_pred = pd.read_csv(lstm_path, index_col='Date', parse_dates=True)
    transformer_pred = pd.read_csv(transformer_path, index_col='Date', parse_dates=True)
    
    # Ensure predictions are aligned
    common_index = lstm_pred.index.intersection(transformer_pred.index)
    lstm_pred = lstm_pred.loc[common_index]
    transformer_pred = transformer_pred.loc[common_index]
    
    return pd.concat([lstm_pred, transformer_pred], axis=1)

def main():
    # Load base model predictions
    predictions = load_predictions('data/processed/lstm_predictions.csv', 'data/processed/transformer_predictions.csv')
    predictions.columns = ['LSTM_pred', 'Transformer_pred']

    # Load actual prices
    actual_prices = pd.read_csv("data/processed/nifty50_processed.csv", index_col="Date", parse_dates=True)
    actual_prices = actual_prices.loc[predictions.index]['Close']

    # Prepare data for ensemble
    X = predictions.values
    y = actual_prices.values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train ensemble
    ensemble = EnsembleModel(base_models=[])  # We don't need base models here as we're using pre-computed predictions
    ensemble.fit(X_train, y_train)

    # Make predictions
    y_pred = ensemble.predict(X_test)

    # Evaluate ensemble
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Ensemble Test MSE: {mse:.4f}")
    print(f"Ensemble Test MAE: {mae:.4f}")

    # Save ensemble model
    joblib.dump(ensemble, 'models/ensemble_model.joblib')

    # Create final predictions DataFrame
    final_predictions = pd.DataFrame({
        'LSTM_pred': predictions['LSTM_pred'],
        'Transformer_pred': predictions['Transformer_pred'],
        'Ensemble_pred': ensemble.predict(X),
        'Actual_Close': actual_prices
    })

    # Save final predictions
    final_predictions.to_csv('data/processed/ensemble_predictions.csv')

    print("Ensemble model training and evaluation complete.")

if __name__ == "__main__":
    main()
