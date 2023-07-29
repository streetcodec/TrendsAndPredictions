import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function to fetch data from Yahoo Finance API
def fetch_stock_data(symbols, start_date, end_date):
    data = yf.download(symbols, start=start_date, end=end_date)
    return data

# Function to calculate moving averages
def calculate_moving_averages(data, window_short=50, window_long=200):
    for symbol in data.columns.levels[1]:
        data['SMA_Short', symbol] = data['Close', symbol].rolling(window=window_short, min_periods=1).mean()
        data['SMA_Long', symbol] = data['Close', symbol].rolling(window=window_long, min_periods=1).mean()
    return data

# Function to calculate Bollinger Bands and RSI
def calculate_bollinger_rsi(data, window=20):
    for symbol in data.columns.levels[1]:
        # Convert 'Open' column to numeric, handling invalid values as NaN
        data['Open', symbol] = pd.to_numeric(data['Open', symbol], errors='coerce')

        # Filter out rows with missing 'Open' data
        data = data.dropna(subset=[('Open', symbol)])

        # Calculate Bollinger Bands
        data['Middle', symbol] = data['Close', symbol].rolling(window=window, min_periods=1).mean()
        data['Std', symbol] = data['Close', symbol].rolling(window=window, min_periods=1).std()
        data['Upper', symbol] = data['Middle', symbol] + 2 * data['Std', symbol]
        data['Lower', symbol] = data['Middle', symbol] - 2 * data['Std', symbol]

        # Calculate RSI
        delta = data['Close', symbol].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        data['RSI', symbol] = rsi

    return data

# Function to visualize market trends with candlestick chart, moving averages, Bollinger Bands, and RSI for each company
def visualize_market_trends(data, symbols):
    num_companies = len(symbols)

    plt.figure(figsize=(12, 6 * num_companies))

    for i, symbol in enumerate(symbols):
        ax1 = plt.subplot(num_companies, 1, i+1)
        ax1.xaxis_date()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.grid(True)
        plt.title(f"{symbol} Market Trends")
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.plot(data.index, data['Open', symbol], 'g', label='Open', linewidth=1)
        plt.plot(data.index, data['Close', symbol], 'b', label='Close', linewidth=1)
        plt.legend()

        ax2 = ax1.twinx()
        ax2.plot(data.index, data['SMA_Short', symbol], 'orange', label='SMA (Short)', linestyle='--')
        ax2.plot(data.index, data['SMA_Long', symbol], 'purple', label='SMA (Long)', linestyle='--')
        ax2.plot(data.index, data['Upper', symbol], 'g', label='Upper Bollinger Band', linestyle='--')
        ax2.plot(data.index, data['Middle', symbol], 'orange', label='Middle Bollinger Band', linestyle='--')
        ax2.plot(data.index, data['Lower', symbol], 'r', label='Lower Bollinger Band', linestyle='--')
        ax2.plot(data.index, data['RSI', symbol], 'm', label='RSI', linestyle='-', alpha=0.7)
        ax2.set_ylabel('Price / RSI')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

def main():
    # Replace with the desired stock symbols and date range
    stock_symbols = ['AAPL', 'MSFT', 'NFLX']
    start_date = '2022-01-01'
    end_date = '2023-01-01'

    # Fetch data from Yahoo Finance API
    stock_data = fetch_stock_data(stock_symbols, start_date, end_date)

    # Calculate moving averages
    stock_data = calculate_moving_averages(stock_data)

    # Calculate Bollinger Bands and RSI
    stock_data = calculate_bollinger_rsi(stock_data)

    # Visualize market trends for each company
    visualize_market_trends(stock_data, stock_symbols)

if __name__ == "__main__":
    main()
