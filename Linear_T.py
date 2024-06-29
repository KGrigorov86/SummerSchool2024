# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 21:28:53 2024

@author: KonstantinGrigorov
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define the file path using double backslashes
file_path = r"C:\Users\KonstantinGrigorov\OneDrive - ROITI Ltd\Документи\Lichni\Summer Academy 2024 Cases\Linear Regression\93_Slovakia_3.xlsx"

# Load the data from the Excel file
data = pd.read_excel(file_path)

# Print the column names to verify
print("Column names in the dataset:")
print(data.columns)

# Filter relevant columns and rename them
# Use exact column names as printed from the above step
data = data[['Т', 'Y (gdp)']]

# Rename columns for easier access
data.columns = ['T', 'Y']

# Convert necessary columns to numeric
data = data.apply(pd.to_numeric, errors='coerce')

# Drop rows with any NaN values
data = data.dropna()

# Filter out rows where T is less than or equal to 0
data = data[data['T'] > 0]

# Split the data into training (T 1 to 68) and testing (T 69 to 78) sets
train_data = data[data['T'] <= 68]
test_data = data[data['T'] > 68]

# Apply linear regression on the training set using T as the independent variable
X_train = train_data[['T']]
y_train = train_data['Y']

model = LinearRegression()
model.fit(X_train, y_train)

# Forecast Y for periods T 69 to 78
X_test = test_data[['T']]
y_test = test_data['Y']
predictions = model.predict(X_test)

# Add predictions to the test data
test_data['predicted_Y'] = predictions

# Calculate R^2 for the forecasted data (T 69 to 78)
r_squared = r2_score(y_test, predictions)

# Print the R^2 value
print(f"R^2 for the forecasted data (T 69 to 78): {r_squared}")

# Combine the training and test data for plotting
train_data['predicted_Y'] = model.predict(train_data[['T']])
combined_data = pd.concat([train_data, test_data])

# Plot the actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(combined_data['T'], combined_data['Y'], label='Actual', color='blue')
plt.plot(combined_data['T'], combined_data['predicted_Y'], label='Predicted', color='red')
plt.title('Linear Regression of Y (gdp) on T')
plt.xlabel('Time Period (T)')
plt.ylabel('Y (gdp)')
plt.legend()
plt.show()
