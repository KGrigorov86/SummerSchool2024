# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 09:07:04 2024

@author: KonstantinGrigorov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score

# Specify the file path
file_path = r'C:\Users\KonstantinGrigorov\OneDrive - ROITI Ltd\Документи\Lichni\Summer Academy 2024 Cases\Crypto\BTC_USDT_1m.csv'

# Read the data from the CSV file
df = pd.read_csv(file_path)

# Ensure the timestamp column is parsed as datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the timestamp column as the index
df.set_index('timestamp', inplace=True)

# Calculate returns based on the close price
btc_returns = df['close'].diff() / df['close'].shift(1)

# Fit ARIMA(5,1,0) model on BTC returns
model = ARIMA(btc_returns, order=(5, 1, 2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=5)

# Make predictions
start_index = 5  # Start at index 5 to account for the lag
end_index = len(btc_returns)
predictions = model_fit.predict(start=start_index, end=end_index-1, dynamic=False)

# Align the lengths of the actual returns and predictions
actual_returns = btc_returns[start_index:]

# Calculate R-squared
r_squared = r2_score(actual_returns, predictions)
print(f'R-squared: {r_squared}')

# Plot the actual returns and the predicted returns
plt.figure(figsize=(12, 6))
plt.plot(actual_returns.index, actual_returns, label='Actual BTC Returns')
plt.plot(actual_returns.index, predictions, label='Predicted BTC Returns', linestyle='--')
plt.title('BTC Returns vs. ARIMA Predictions')
plt.xlabel('Timestamp')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.show()


