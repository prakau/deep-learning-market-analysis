import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_stock_data(symbol, start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def fetch_index_data(index_symbol, start_date, end_date):
    """Fetch index data from Yahoo Finance."""
    return fetch_stock_data(index_symbol, start_date, end_date)

def fetch_economic_indicators(start_date, end_date):
    """Fetch economic indicators from RBI website."""
    # This is a placeholder. In a real scenario, you'd need to implement web scraping or use an API.
    # For demonstration purposes, we'll create some dummy data
    date_range = pd.date_range(start=start_date, end=end_date, freq='M')
    data = {
        'Date': date_range,
        'GDP_Growth': np.random.normal(7, 1, len(date_range)),
        'Inflation_Rate': np.random.normal(4, 0.5, len(date_range)),
        'Interest_Rate': np.random.normal(6, 0.3, len(date_range))
    }
    return pd.DataFrame(data).set_index('Date')

def fetch_all_data(symbols, indices, start_date, end_date):
    """Fetch all required data concurrently."""
    with ThreadPoolExecutor() as executor:
        stock_data = list(executor.map(partial(fetch_stock_data, start_date=start_date, end_date=end_date), symbols))
        index_data = list(executor.map(partial(fetch_index_data, start_date=start_date, end_date=end_date), indices))
    
    economic_data = fetch_economic_indicators(start_date, end_date)
    
    return stock_data, index_data, economic_data

def save_data(data, filename, directory='data/raw'):
    """Save the fetched data to a CSV file."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{filename}.csv")
    data.to_csv(filepath)
    logger.info(f"Data saved to {filepath}")

def main(start_date, end_date):
    # Define symbols and indices
    symbols = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS']
    indices = ['^NSEI', '^BSESN']  # Nifty 50 and Sensex
    
    logger.info("Starting data collection...")
    stock_data, index_data, economic_data = fetch_all_data(symbols, indices, start_date, end_date)
    
    # Save stock data
    for symbol, data in zip(symbols, stock_data):
        if data is not None:
            save_data(data, f"{symbol}_historical")
    
    # Save index data
    for index, data in zip(indices, index_data):
        if data is not None:
            save_data(data, f"{index}_historical")
    
    # Save economic data
    save_data(economic_data, "economic_indicators")
    
    logger.info("Data collection complete.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect market data for Indian stocks and indices.")
    parser.add_argument('--start-date', type=str, default='2010-01-01', help='Start date for data collection (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date for data collection (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    main(args.start_date, args.end_date)
