import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class MarketVisualizer:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, index_col='Date', parse_dates=True)

    def plot_price_trends(self, save_path=None):
        plt.figure(figsize=(15, 8))
        plt.plot(self.data.index, self.data['Actual_Close'], label='Actual Close')
        plt.plot(self.data.index, self.data['LSTM_pred'], label='LSTM Prediction')
        plt.plot(self.data.index, self.data['Transformer_pred'], label='Transformer Prediction')
        plt.plot(self.data.index, self.data['Ensemble_pred'], label='Ensemble Prediction')
        plt.title('Market Price Trends and Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_prediction_errors(self, save_path=None):
        errors = pd.DataFrame({
            'LSTM Error': self.data['Actual_Close'] - self.data['LSTM_pred'],
            'Transformer Error': self.data['Actual_Close'] - self.data['Transformer_pred'],
            'Ensemble Error': self.data['Actual_Close'] - self.data['Ensemble_pred']
        })
        
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=errors)
        plt.title('Prediction Errors by Model')
        plt.ylabel('Error')
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def create_interactive_dashboard(self, save_path=None):
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Price Trends and Predictions', 'Prediction Errors',
                                            'Error Distribution', 'Model Performance Comparison'))

        # Price Trends and Predictions
        for col in ['Actual_Close', 'LSTM_pred', 'Transformer_pred', 'Ensemble_pred']:
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data[col], name=col), row=1, col=1)

        # Prediction Errors
        for col in ['LSTM_pred', 'Transformer_pred', 'Ensemble_pred']:
            fig.add_trace(go.Scatter(x=self.data.index, y=self.data['Actual_Close'] - self.data[col],
                                     name=f'{col} Error'), row=1, col=2)

        # Error Distribution
        for col in ['LSTM_pred', 'Transformer_pred', 'Ensemble_pred']:
            fig.add_trace(go.Histogram(x=self.data['Actual_Close'] - self.data[col],
                                       name=f'{col} Error Distribution'), row=2, col=1)

        # Model Performance Comparison
        mse = {col: ((self.data['Actual_Close'] - self.data[col])**2).mean() for col in ['LSTM_pred', 'Transformer_pred', 'Ensemble_pred']}
        fig.add_trace(go.Bar(x=list(mse.keys()), y=list(mse.values()), name='MSE'), row=2, col=2)

        fig.update_layout(height=1000, width=1500, title_text="Market Forecasting Dashboard")
        if save_path:
            fig.write_html(save_path)
        fig.show()

def main():
    visualizer = MarketVisualizer('data/processed/ensemble_predictions.csv')
    visualizer.plot_price_trends('visualizations/price_trends.png')
    visualizer.plot_prediction_errors('visualizations/prediction_errors.png')
    visualizer.create_interactive_dashboard('visualizations/interactive_dashboard.html')

if __name__ == "__main__":
    main()
