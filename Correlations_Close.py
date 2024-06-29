# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:59:04 2024

@author: KonstantinGrigorov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score

# Specify the file path
file_path = r'C:\Users\KonstantinGrigorov\OneDrive - ROITI Ltd\Документи\Lichni\Summer Academy 2024 Cases\Crypto\BTC_USDT_1m.xlsx'

# Read the data from the Excel file
df = pd.read_excel(file_path)

# Ensure the timestamp column is parsed as datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set the timestamp column as the index
df.set_index('timestamp', inplace=True)

# Calculate returns based on the close price
btc_returns = df['close'].pct_change().dropna()

# Fit ARIMA(5,1,0) model on BTC returns
model = ARIMA(btc_returns, order=(5, 1, 0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=1, end=len(btc_returns), dynamic=False)

# Calculate R-squared
r_squared = r2_score(btc_returns[1:], predictions)
print(f'R-squared: {r_squared}')

# Plot the actual returns and the predicted returns
plt.figure(figsize=(12, 6))
plt.plot(btc_returns, label='Actual BTC Returns')
plt.plot(predictions, label='Predicted BTC Returns', linestyle='--')
plt.title('BTC Returns vs. ARIMA Predictions')
plt.xlabel('Timestamp')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.show()
