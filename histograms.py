"""

IN THE REPORT IGNORE ID AND NAME HISTOGRAMS

WE IGNORE THE tests.csv FILE AS IT DOES NOT HAVE LABELS

"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the dataset
df = pd.read_csv('combats.csv')  # pokemons.csv / combats.csv

# Create a folder to save plots
output_folder = 'eda_plots_combats'  # eda_plots for pokemons.csv, eda_plots_combats for combats.csv
os.makedirs(output_folder, exist_ok=True)

# Identify numeric and categorical columns
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object', 'bool']).columns

# Plot histograms for numeric columns
for col in num_cols:
    plt.figure(figsize=(8, 6))
    df[col].hist(bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{col}_hist.png')
    plt.close()

# Plot bar charts for categorical columns
for col in cat_cols:
    plt.figure(figsize=(10, 6))
    df[col].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
    plt.title(f'Bar Chart: {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/{col}_bar.png')
    plt.close()



print("Plots saved in folder:", output_folder)
