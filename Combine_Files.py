# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:59:04 2024

@author: KonstantinGrigorov
"""
import pandas as pd
import os

# Specify the directory containing the files
directory = r'C:\Users\KonstantinGrigorov\OneDrive - ROITI Ltd\Документи\Lichni\Summer Academy 2024 Cases\Crypto\data'

# Initialize an empty list to hold the DataFrames
dataframes = []

# Loop through the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.csv'):  # Check if the file is a CSV file
        file_path = os.path.join(directory, filename)
        prefix = filename.split('_USDT_1m')[0]  # Extract the prefix from the filename
        df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
        df['Crypto'] = prefix  # Add the prefix as a new column named 'Crypto'
        dataframes.append(df)  # Append the DataFrame to the list

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Reorder the columns to have 'timestamp' and 'Crypto' as the first two columns
columns = ['timestamp', 'Crypto'] + [col for col in combined_df.columns if col not in ['timestamp', 'Crypto']]
combined_df = combined_df[columns]

# Display the combined DataFrame
print(combined_df)

# Assuming 'df' is your combined DataFrame with a 'Crypto' column

# Split the DataFrame based on the 'Crypto' column
crypto_groups = df.groupby('Crypto')

# Store the DataFrames in a dictionary
crypto_dataframes = {crypto: group for crypto, group in crypto_groups}


