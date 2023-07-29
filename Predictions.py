import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Can use other valid symbols as well.
symbol = "AAPL"
start_date = "2015-01-01"
end_date = "2021-12-31"
# Fetch historical stock data.
stock_data = yf.download(symbol, start=start_date, end=end_date)

# Display historical stock price for Apple Inc. (AAPL)
print("Historical Stock Price:")
print(stock_data)

# Prepare data for training
X = np.arange(len(stock_data)).reshape(-1, 1)
y = stock_data['Close'].values

# Training a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predicting the future closing price using the trained model
num_days_into_future = 90
future_dates = pd.date_range(start=stock_data.index[-1], periods=num_days_into_future + 1, closed='right')
future_X = np.arange(len(stock_data), len(stock_data) + num_days_into_future).reshape(-1, 1)
future_y_pred = model.predict(future_X)

# Convert predicted prices to a DataFrame
future_predictions = pd.DataFrame({'Date': future_dates[1:], 'Predicted_Close': future_y_pred[1:]})

# Display the predicted future closing prices
print("\nPredicted Future Closing Prices:")
print(future_predictions)

# Plot historical and predicted future closing prices
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Historical Close')
plt.plot(future_predictions['Date'], future_predictions['Predicted_Close'], label='Predicted Close', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Historical and Predicted Future Closing Prices')
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()
