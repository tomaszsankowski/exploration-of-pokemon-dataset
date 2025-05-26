"""
Descriptive Statistics and Missing Data Analysis
For Pokemon and Combats datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs('descriptive_analysis', exist_ok=True)


def analyze_pokemon_dataset():
    """Analyze pokemon.csv dataset"""
    print("=" * 60)
    print("POKEMON DATASET ANALYSIS")
    print("=" * 60)

    # Load dataset
    pokemon_df = pd.read_csv('pokemon.csv')

    print(f"Dataset shape: {pokemon_df.shape}")
    print(f"Number of rows: {pokemon_df.shape[0]}")
    print(f"Number of columns: {pokemon_df.shape[1]}")
    print("\n")

    # Basic info
    print("DATASET INFO:")
    print("-" * 40)
    pokemon_df.info()
    print("\n")

    # Identify numeric and categorical columns
    numeric_cols = pokemon_df.select_dtypes(
        include=[np.number]).columns.tolist()
    categorical_cols = pokemon_df.select_dtypes(
        include=['object', 'bool']).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print("\n")

    # DESCRIPTIVE STATISTICS FOR NUMERIC ATTRIBUTES
    print("DESCRIPTIVE STATISTICS FOR NUMERIC ATTRIBUTES:")
    print("=" * 50)

    numeric_stats = pokemon_df[numeric_cols].describe()
    print(numeric_stats)
    print("\n")

    # Additional statistics
    print("ADDITIONAL NUMERIC STATISTICS:")
    print("-" * 40)
    for col in numeric_cols:
        data = pokemon_df[col]
        print(f"\n{col}:")
        print(f"  Mean: {data.mean():.2f}")
        print(f"  Median: {data.median():.2f}")
        print(f"  Min: {data.min()}")
        print(f"  Max: {data.max()}")
        print(f"  Q1 (25%): {data.quantile(0.25):.2f}")
        print(f"  Q3 (75%): {data.quantile(0.75):.2f}")
        print(f"  Standard Deviation: {data.std():.2f}")
        print(f"  Variance: {data.var():.2f}")
        print(f"  Skewness: {data.skew():.2f}")
        print(f"  Kurtosis: {data.kurtosis():.2f}")

    # DESCRIPTIVE STATISTICS FOR CATEGORICAL ATTRIBUTES
    print("\n\nDESCRIPTIVE STATISTICS FOR CATEGORICAL ATTRIBUTES:")
    print("=" * 55)

    for col in categorical_cols:
        print(f"\n{col}:")
        print("-" * 30)
        value_counts = pokemon_df[col].value_counts(dropna=False)
        percentages = pokemon_df[col].value_counts(
            normalize=True, dropna=False) * 100

        print("Value counts:")
        for value, count in value_counts.items():
            percentage = percentages[value]
            print(f"  {value}: {count} ({percentage:.2f}%)")

        print(f"Total unique values: {pokemon_df[col].nunique()}")
        print(
            f"Most common value: {pokemon_df[col].mode().iloc[0] if not pokemon_df[col].mode().empty else 'N/A'}")

    # MISSING DATA ANALYSIS
    print("\n\nMISSING DATA ANALYSIS:")
    print("=" * 30)

    missing_data = pokemon_df.isnull().sum()
    missing_percentage = (pokemon_df.isnull().sum() / len(pokemon_df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percentage.values
    })

    print(missing_df)
    print(f"\nTotal missing values: {pokemon_df.isnull().sum().sum()}")
    print(f"Columns with missing data: {(missing_data > 0).sum()}")

    # Visualize missing data
    plt.figure(figsize=(12, 8))

    # Missing data heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(pokemon_df.isnull(), cbar=True,
                yticklabels=False, cmap='viridis')
    plt.title('Missing Data Heatmap - Pokemon Dataset')
    plt.xlabel('Columns')

    # Missing data bar plot
    plt.subplot(2, 2, 2)
    missing_data[missing_data > 0].plot(kind='bar', color='red', alpha=0.7)
    plt.title('Missing Data Count by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Count')
    plt.xticks(rotation=45)

    # Missing data percentage
    plt.subplot(2, 2, 3)
    missing_percentage[missing_percentage > 0].plot(
        kind='bar', color='orange', alpha=0.7)
    plt.title('Missing Data Percentage by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Percentage (%)')
    plt.xticks(rotation=45)

    # Correlation matrix of missing data
    plt.subplot(2, 2, 4)
    missing_corr = pokemon_df.isnull().corr()
    sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Missing Data Correlation')

    plt.tight_layout()
    plt.savefig('descriptive_analysis/pokemon_missing_data_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed statistics to CSV
    numeric_stats.to_csv('descriptive_analysis/pokemon_numeric_stats.csv')
    missing_df.to_csv(
        'descriptive_analysis/pokemon_missing_data.csv', index=False)

    return pokemon_df


def analyze_combats_dataset():
    """Analyze combats.csv dataset"""
    print("\n\n" + "=" * 60)
    print("COMBATS DATASET ANALYSIS")
    print("=" * 60)

    # Load dataset
    combats_df = pd.read_csv('combats.csv')

    print(f"Dataset shape: {combats_df.shape}")
    print(f"Number of rows: {combats_df.shape[0]}")
    print(f"Number of columns: {combats_df.shape[1]}")
    print("\n")

    # Basic info
    print("DATASET INFO:")
    print("-" * 40)
    combats_df.info()
    print("\n")

    # Identify numeric and categorical columns
    numeric_cols = combats_df.select_dtypes(
        include=[np.number]).columns.tolist()
    categorical_cols = combats_df.select_dtypes(
        include=['object', 'bool']).columns.tolist()

    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")
    print("\n")

    # DESCRIPTIVE STATISTICS FOR NUMERIC ATTRIBUTES
    print("DESCRIPTIVE STATISTICS FOR NUMERIC ATTRIBUTES:")
    print("=" * 50)

    if numeric_cols:
        numeric_stats = combats_df[numeric_cols].describe()
        print(numeric_stats)
        print("\n")

        # Additional statistics
        print("ADDITIONAL NUMERIC STATISTICS:")
        print("-" * 40)
        for col in numeric_cols:
            data = combats_df[col]
            print(f"\n{col}:")
            print(f"  Mean: {data.mean():.2f}")
            print(f"  Median: {data.median():.2f}")
            print(f"  Min: {data.min()}")
            print(f"  Max: {data.max()}")
            print(f"  Q1 (25%): {data.quantile(0.25):.2f}")
            print(f"  Q3 (75%): {data.quantile(0.75):.2f}")
            print(f"  Standard Deviation: {data.std():.2f}")
            print(f"  Variance: {data.var():.2f}")
            print(f"  Skewness: {data.skew():.2f}")
            print(f"  Kurtosis: {data.kurtosis():.2f}")

    # COMBAT-SPECIFIC ANALYSIS
    print("\n\nCOMBAT-SPECIFIC ANALYSIS:")
    print("=" * 30)

    # Winner analysis
    first_wins = (combats_df['Winner'] == combats_df['First_pokemon']).sum()
    second_wins = (combats_df['Winner'] == combats_df['Second_pokemon']).sum()
    total_combats = len(combats_df)

    print(
        f"First Pokemon wins: {first_wins} ({first_wins/total_combats*100:.2f}%)")
    print(
        f"Second Pokemon wins: {second_wins} ({second_wins/total_combats*100:.2f}%)")
    print(f"Total combats: {total_combats}")

    # Most common Pokemon in battles
    print("\nMost frequently battling Pokemon:")
    all_pokemon_battles = pd.concat(
        [combats_df['First_pokemon'], combats_df['Second_pokemon']])
    most_common_battlers = all_pokemon_battles.value_counts().head(10)
    print(most_common_battlers)

    # Most successful Pokemon
    print("\nMost successful Pokemon (by wins):")
    winner_counts = combats_df['Winner'].value_counts().head(10)
    print(winner_counts)

    # DESCRIPTIVE STATISTICS FOR CATEGORICAL ATTRIBUTES
    if categorical_cols:
        print("\n\nDESCRIPTIVE STATISTICS FOR CATEGORICAL ATTRIBUTES:")
        print("=" * 55)

        for col in categorical_cols:
            print(f"\n{col}:")
            print("-" * 30)
            value_counts = combats_df[col].value_counts(dropna=False)
            percentages = combats_df[col].value_counts(
                normalize=True, dropna=False) * 100

            print("Value counts:")
            for value, count in value_counts.items():
                percentage = percentages[value]
                print(f"  {value}: {count} ({percentage:.2f}%)")

            print(f"Total unique values: {combats_df[col].nunique()}")

    # MISSING DATA ANALYSIS
    print("\n\nMISSING DATA ANALYSIS:")
    print("=" * 30)

    missing_data = combats_df.isnull().sum()
    missing_percentage = (combats_df.isnull().sum() / len(combats_df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percentage.values
    })

    print(missing_df)
    print(f"\nTotal missing values: {combats_df.isnull().sum().sum()}")
    print(f"Columns with missing data: {(missing_data > 0).sum()}")

    # Visualize missing data and combat statistics
    plt.figure(figsize=(15, 10))

    # Missing data heatmap
    plt.subplot(2, 3, 1)
    if combats_df.isnull().sum().sum() > 0:
        sns.heatmap(combats_df.isnull(), cbar=True,
                    yticklabels=False, cmap='viridis')
        plt.title('Missing Data Heatmap - Combats Dataset')
    else:
        plt.text(0.5, 0.5, 'No Missing Data',
                 ha='center', va='center', fontsize=16)
        plt.title('Missing Data - Combats Dataset')
    plt.xlabel('Columns')

    # Winner distribution
    plt.subplot(2, 3, 2)
    winner_data = ['First Pokemon', 'Second Pokemon']
    winner_counts_plot = [first_wins, second_wins]
    plt.bar(winner_data, winner_counts_plot,
            color=['blue', 'orange'], alpha=0.7)
    plt.title('Winner Distribution')
    plt.ylabel('Number of Wins')

    # Pokemon ID distributions
    plt.subplot(2, 3, 3)
    plt.hist(combats_df['First_pokemon'], bins=50,
             alpha=0.5, label='First Pokemon', color='blue')
    plt.hist(combats_df['Second_pokemon'], bins=50,
             alpha=0.5, label='Second Pokemon', color='orange')
    plt.title('Pokemon ID Distribution in Battles')
    plt.xlabel('Pokemon ID')
    plt.ylabel('Frequency')
    plt.legend()

    # Winner ID distribution
    plt.subplot(2, 3, 4)
    plt.hist(combats_df['Winner'], bins=50, alpha=0.7, color='green')
    plt.title('Winner Pokemon ID Distribution')
    plt.xlabel('Pokemon ID')
    plt.ylabel('Frequency')

    # Most common battlers
    plt.subplot(2, 3, 5)
    most_common_battlers.head(10).plot(kind='bar', color='purple', alpha=0.7)
    plt.title('Top 10 Most Frequently Battling Pokemon')
    plt.xlabel('Pokemon ID')
    plt.ylabel('Battle Count')
    plt.xticks(rotation=45)

    # Most successful Pokemon
    plt.subplot(2, 3, 6)
    winner_counts.head(10).plot(kind='bar', color='red', alpha=0.7)
    plt.title('Top 10 Most Successful Pokemon')
    plt.xlabel('Pokemon ID')
    plt.ylabel('Win Count')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('descriptive_analysis/combats_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Save detailed statistics to CSV
    if numeric_cols:
        numeric_stats.to_csv('descriptive_analysis/combats_numeric_stats.csv')
    missing_df.to_csv(
        'descriptive_analysis/combats_missing_data.csv', index=False)

    # Save battle statistics
    battle_stats = pd.DataFrame({
        'Metric': ['First Pokemon Wins', 'Second Pokemon Wins', 'Total Combats', 'First Win Rate (%)', 'Second Win Rate (%)'],
        'Value': [first_wins, second_wins, total_combats, first_wins/total_combats*100, second_wins/total_combats*100]
    })
    battle_stats.to_csv(
        'descriptive_analysis/battle_statistics.csv', index=False)

    return combats_df


def create_combined_analysis(pokemon_df, combats_df):
    """Create combined analysis of both datasets"""
    print("\n\n" + "=" * 60)
    print("COMBINED DATASET ANALYSIS")
    print("=" * 60)

    # Merge datasets for combined analysis
    pokemon_battles = combats_df.merge(
        pokemon_df.add_suffix('_first'),
        left_on='First_pokemon',
        right_on='ID_first'
    ).merge(
        pokemon_df.add_suffix('_second'),
        left_on='Second_pokemon',
        right_on='ID_second'
    )

    print(f"Combined dataset shape: {pokemon_battles.shape}")

    # Analysis of legendary Pokemon in battles
    legendary_first = pokemon_battles['Legendary_first'].sum()
    legendary_second = pokemon_battles['Legendary_second'].sum()
    legendary_winners = pokemon_battles[pokemon_battles['Winner'] == pokemon_battles['First_pokemon']]['Legendary_first'].sum() + \
        pokemon_battles[pokemon_battles['Winner'] ==
                        pokemon_battles['Second_pokemon']]['Legendary_second'].sum()

    print(f"\nLegendary Pokemon Analysis:")
    print(f"Battles with legendary as first Pokemon: {legendary_first}")
    print(f"Battles with legendary as second Pokemon: {legendary_second}")
    print(f"Battles won by legendary Pokemon: {legendary_winners}")

    # Type analysis in battles
    print(f"\nType Analysis in Battles:")
    type1_first_counts = pokemon_battles['Type 1_first'].value_counts().head(5)
    print("Most common first Pokemon types:")
    print(type1_first_counts)

    # Save combined analysis
    summary_stats = {
        'Pokemon Dataset Shape': pokemon_df.shape,
        'Combats Dataset Shape': combats_df.shape,
        'Combined Dataset Shape': pokemon_battles.shape,
        'Total Unique Pokemon': pokemon_df['ID'].nunique(),
        'Pokemon in Battles': len(set(combats_df['First_pokemon'].tolist() + combats_df['Second_pokemon'].tolist())),
        'Legendary in Battles': legendary_first + legendary_second,
        'Legendary Wins': legendary_winners
    }

    with open('descriptive_analysis/combined_summary.txt', 'w') as f:
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    print("Starting Descriptive Statistics and Missing Data Analysis...")
    print("This analysis includes:")
    print("- Summary statistics (mean, median, min, max, quartiles, std dev)")
    print("- Frequency counts for categorical variables")
    print("- Missing data analysis and visualization")
    print("- Combined dataset insights")
    print("\n")

    # Analyze both datasets
    pokemon_df = analyze_pokemon_dataset()
    combats_df = analyze_combats_dataset()
    create_combined_analysis(pokemon_df, combats_df)

    print(f"\n\nAnalysis completed! Results saved in 'descriptive_analysis' folder:")
    print("- pokemon_missing_data_analysis.png")
    print("- combats_analysis.png")
    print("- pokemon_numeric_stats.csv")
    print("- combats_numeric_stats.csv")
    print("- pokemon_missing_data.csv")
    print("- combats_missing_data.csv")
    print("- battle_statistics.csv")
    print("- combined_summary.txt")
