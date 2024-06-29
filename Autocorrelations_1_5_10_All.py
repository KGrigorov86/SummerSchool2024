# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 20:46:21 2024

@author: KonstantinGrigorov
"""

import pandas as pd
import os

# Specify the directory containing the files
directory = r'C:\Users\KonstantinGrigorov\OneDrive - ROITI Ltd\Документи\Lichni\Summer Academy 2024 Cases\Crypto\data'

# Initialize an empty list to hold the DataFrames
dataframes = []

# Function to calculate daily returns as the difference between the next and the previous close value divided by the previous one
def calculate_daily_returns(df):
    df['daily_return'] = df['close'].diff() / df['close'].shift(1)
    return df

# Function to calculate autocorrelation for a range of lags
def calculate_autocorrelations(df, max_lag):
    autocorrelations = {}
    for lag in range(1, max_lag + 1):
        autocorrelations[f'autocorr_{lag}'] = df['daily_return'].autocorr(lag)
    return autocorrelations

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # Check if the file is a CSV file
        file_path = os.path.join(directory, filename)
        prefix = filename.split('_USDT_1m')[0]  # Extract the prefix from the filename
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Calculate daily returns
        df = calculate_daily_returns(df)
        
        # Calculate autocorrelations for lags from 1 to 10
        autocorrelations = calculate_autocorrelations(df, 10)
        
        # Create a DataFrame from the autocorrelations
        autocorr_df = pd.DataFrame(autocorrelations, index=[0])
        autocorr_df['Crypto'] = prefix  # Add the prefix as a new column named 'Crypto'
        
        # Append the autocorr_df to the list
        dataframes.append(autocorr_df)

# Combine all autocorrelation DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Split the combined DataFrame based on the 'Crypto' column
crypto_groups = combined_df.groupby('Crypto')

# Store the DataFrames in a dictionary
crypto_dataframes = {crypto: group for crypto, group in crypto_groups}
