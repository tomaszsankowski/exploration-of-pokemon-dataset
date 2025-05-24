import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory
os.makedirs('correlation_plots', exist_ok=True)

try:
    # Load the dataset
    pokemon_df = pd.read_csv('pokemon.csv')

    # Select only numeric columns (including 'Legendary' as boolean)
    selected_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']
    df_numeric = pokemon_df[selected_cols].copy()

    # Convert boolean to integer (if not already)
    if df_numeric['Legendary'].dtype == bool:
        df_numeric['Legendary'] = df_numeric['Legendary'].astype(int)

    # Compute correlation matrix
    corr_matrix = df_numeric.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Numeric Correlation Matrix - pokemon.csv')
    plt.tight_layout()
    plt.savefig('correlation_plots/pokemon_correlation.png')
    plt.close()
    print("Saved correlation heatmap for pokemon.csv")

except Exception as e:
    print(f"Error processing pokemon.csv: {e}")
