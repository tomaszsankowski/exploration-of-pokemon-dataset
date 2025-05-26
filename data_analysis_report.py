\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create a directory for saving plots
output_dir = 'analysis_report_plots'
os.makedirs(output_dir, exist_ok=True)

# Load datasets
pokemon_df = pd.read_csv('pokemon.csv')
combats_df = pd.read_csv('combats.csv')
# tests_df = pd.read_csv('tests.csv') # As per README, tests.csv is for performance evaluation and lacks winner labels

print("Pokemon DataFrame Head:")
print(pokemon_df.head())
print("\\nCombats DataFrame Head:")
print(combats_df.head())

# --- 1. Descriptive Statistics ---
print("\\n--- 1. Descriptive Statistics ---")

# Numeric attributes summary from pokemon.csv
print("\\nDescriptive statistics for numeric attributes (pokemon.csv):")
numeric_cols_pokemon = pokemon_df.select_dtypes(
    include=np.number).columns.tolist()
# Exclude 'ID' and 'Generation' if they are not to be treated as continuous variables for stats like mean/median in this context
# However, the user asked for all numeric attributes. 'Generation' can be debated, but 'ID' is clearly an identifier.
# For now, including all numeric as per request, but will exclude ID from plots.
numeric_summary = pokemon_df[numeric_cols_pokemon].describe()
print(numeric_summary)
numeric_summary.to_csv(os.path.join(output_dir, 'pokemon_numeric_summary.csv'))

# Nominal attributes summary from pokemon.csv
print("\\nDescriptive statistics for nominal attributes (pokemon.csv):")
nominal_cols_pokemon = ['Type 1', 'Type 2', 'Legendary']
for col in nominal_cols_pokemon:
    print(f"\\nFrequency and percentage for {col}:")
    count = pokemon_df[col].value_counts(dropna=False)
    percentage = pokemon_df[col].value_counts(
        dropna=False, normalize=True) * 100
    nominal_summary_col = pd.DataFrame(
        {'Count': count, 'Percentage': percentage})
    print(nominal_summary_col)
    nominal_summary_col.to_csv(os.path.join(
        output_dir, f'pokemon_nominal_{col.replace(" ", "_")}_summary.csv'))

    # Plot for nominal attributes
    plt.figure(figsize=(10, 6))
    # sns.countplot(y=pokemon_df[col], order=pokemon_df[col].value_counts(
    #     dropna=False).index, palette='viridis') # Old line with FutureWarning
    sns.countplot(y=pokemon_df[col], order=pokemon_df[col].value_counts(
        dropna=False).index, color='skyblue', hue=None) # Using 'color' for uniform coloring, or specify a list if needed.
                                                        # If 'viridis' was meant to apply to different bars,
                                                        # and 'y' is the categorical variable, seaborn handles this by default
                                                        # or 'palette' can be used if 'hue' is also specified.
                                                        # For a simple countplot without hue, 'color' is more direct.
                                                        # Alternatively, to mimic 'viridis' behavior across bars:
                                                        # sns.countplot(y=pokemon_df[col], order=pokemon_df[col].value_counts(dropna=False).index, hue=pokemon_df[col], palette='viridis', legend=False)
                                                        # For simplicity, using a single color.
    plt.title(f'Distribution of Pokemon {col}')
    plt.xlabel('Count')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'pokemon_{col.replace(" ", "_")}_distribution.png'))
    plt.close()

# --- 2. Missing Data Analysis ---
print("\\n--- 2. Missing Data Analysis ---")
print("\\nMissing data in pokemon.csv:")
missing_pokemon = pokemon_df.isnull().sum()
missing_pokemon_percent = (pokemon_df.isnull().sum() / len(pokemon_df)) * 100
missing_pokemon_df = pd.DataFrame(
    {'Missing Count': missing_pokemon, 'Missing Percentage': missing_pokemon_percent})
print(missing_pokemon_df[missing_pokemon_df['Missing Count'] > 0])
missing_pokemon_df.to_csv(os.path.join(
    output_dir, 'pokemon_missing_data_summary.csv'))

print("\\nMissing data in combats.csv:")
missing_combats = combats_df.isnull().sum()
missing_combats_percent = (combats_df.isnull().sum() / len(combats_df)) * 100
missing_combats_df = pd.DataFrame(
    {'Missing Count': missing_combats, 'Missing Percentage': missing_combats_percent})
# Should be none based on typical Kaggle datasets of this type
print(missing_combats_df[missing_combats_df['Missing Count'] > 0])
missing_combats_df.to_csv(os.path.join(
    output_dir, 'combats_missing_data_summary.csv'))


# Heatmap for missing values in pokemon.csv
plt.figure(figsize=(10, 6))
sns.heatmap(pokemon_df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Missing Data Heatmap - pokemon.csv')
plt.savefig(os.path.join(output_dir, 'pokemon_missing_data_heatmap.png'))
plt.close()

# --- 3. Target Analysis (Class Distribution) ---
print("\\n--- 3. Target Analysis (Class Distribution) ---")

# Win percentage of First Pokemon vs Second Pokemon
first_wins = (combats_df['Winner'] == combats_df['First_pokemon']).sum()
second_wins = (combats_df['Winner'] == combats_df['Second_pokemon']).sum()
total_combats = len(combats_df)

first_win_rate = (first_wins / total_combats) * 100
second_win_rate = (second_wins / total_combats) * 100

print(f"Total Combats: {total_combats}")
print(f"First Pokemon Wins: {first_wins} ({first_win_rate:.2f}%)")
print(f"Second Pokemon Wins: {second_wins} ({second_win_rate:.2f}%)")

# Plot for win distribution
plt.figure(figsize=(8, 6))
# sns.barplot(x=['First Pokemon Wins', 'Second Pokemon Wins'], y=[
#             first_wins, second_wins], palette=['#3498db', '#e74c3c']) # Old line
sns.barplot(x=['First Pokemon Wins', 'Second Pokemon Wins'], y=[
            first_wins, second_wins], hue=['First Pokemon Wins', 'Second Pokemon Wins'], palette=['#3498db', '#e74c3c'], legend=False) # Corrected: Added hue
plt.title('Combat Outcome Distribution (Win Counts)')
plt.ylabel('Number of Wins')
for i, v in enumerate([first_wins, second_wins]):
    plt.text(i, v + 500, str(v) + f"\\n({[first_win_rate, second_win_rate][i]:.2f}%)",
             color='black', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'combat_win_distribution.png'))
plt.close()

# Distribution of Legendary Pokemon
print("\\nLegendary Pokemon Distribution:")
legendary_counts = pokemon_df['Legendary'].value_counts()
legendary_percentage = pokemon_df['Legendary'].value_counts(
    normalize=True) * 100
legendary_dist_df = pd.DataFrame(
    {'Count': legendary_counts, 'Percentage': legendary_percentage})
print(legendary_dist_df)
legendary_dist_df.to_csv(os.path.join(
    output_dir, 'pokemon_legendary_distribution.csv'))

plt.figure(figsize=(6, 6))
legendary_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'gold'], startangle=90,
                      wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 12})
plt.title('Distribution of Legendary Pokemon')
plt.ylabel('')  # Hide the 'Legendary' ylabel from pie chart
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pokemon_legendary_distribution_pie.png'))
plt.close()

# --- 4. Comparison of Features based on Combat Outcome ---
print("\\n--- 4. Comparison of Features based on Combat Outcome ---")

# Merge combat data with Pokemon stats
# We need to merge twice: once for the winner's stats, and then determine the loser to get their stats.

# Add a 'Loser' column to combats_df
combats_df['Loser'] = np.where(combats_df['Winner'] == combats_df['First_pokemon'],
                               combats_df['Second_pokemon'],
                               combats_df['First_pokemon'])

# Merge to get Winner stats
winner_stats = combats_df.merge(pokemon_df.add_prefix(
    'Winner_'), left_on='Winner', right_on='Winner_ID')
# Merge to get Loser stats
combat_full_stats = winner_stats.merge(pokemon_df.add_prefix(
    'Loser_'), left_on='Loser', right_on='Loser_ID')

# Select relevant stat columns for comparison
stat_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
winner_stat_cols = ['Winner_' + col for col in stat_cols]
loser_stat_cols = ['Loser_' + col for col in stat_cols]

# Calculate average stats for winners and losers
avg_winner_stats = combat_full_stats[winner_stat_cols].mean().rename(
    lambda x: x.replace('Winner_', ''))
avg_loser_stats = combat_full_stats[loser_stat_cols].mean().rename(
    lambda x: x.replace('Loser_', ''))

comparison_stats_df = pd.DataFrame(
    {'Average_Winner_Stats': avg_winner_stats, 'Average_Loser_Stats': avg_loser_stats})
print("\\nAverage stats for Winning vs. Losing Pokemon:")
print(comparison_stats_df)
comparison_stats_df.to_csv(os.path.join(
    output_dir, 'pokemon_winner_vs_loser_avg_stats.csv'))

# Plotting average stats comparison
comparison_stats_df.plot(kind='bar', figsize=(12, 7), colormap='coolwarm_r')
plt.title('Average Stats: Winner vs. Loser Pokemon')
plt.ylabel('Average Stat Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(
    output_dir, 'pokemon_avg_stats_winner_vs_loser_bar.png'))
plt.close()

# Boxplots/Violin plots for each stat
print("\\nGenerating boxplots for Winner vs Loser stats...")
for stat in stat_cols:
    plt.figure(figsize=(8, 6))
    # Create a temporary DataFrame for plotting this specific stat
    plot_df = pd.DataFrame({
        'Stat_Value': pd.concat([combat_full_stats['Winner_' + stat], combat_full_stats['Loser_' + stat]], ignore_index=True),
        'Outcome': ['Winner'] * len(combat_full_stats) + ['Loser'] * len(combat_full_stats)
    })
    # sns.boxplot(x='Outcome', y='Stat_Value', data=plot_df, palette={
    #             'Winner': 'lightgreen', 'Loser': 'salmon'}) # Old line
    sns.boxplot(x='Outcome', y='Stat_Value', hue='Outcome', data=plot_df, palette={
                'Winner': 'lightgreen', 'Loser': 'salmon'}, legend=False) # Corrected: Added hue and legend=False
    plt.title(f'{stat} Distribution: Winner vs. Loser')
    plt.ylabel(stat)
    plt.xlabel('Combat Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'pokemon_{stat.replace(" ", "_")}_boxplot_winner_vs_loser.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    # sns.violinplot(x='Outcome', y='Stat_Value', data=plot_df, palette={
    #                'Winner': 'lightgreen', 'Loser': 'salmon'}) # Old line
    sns.violinplot(x='Outcome', y='Stat_Value', hue='Outcome', data=plot_df, palette={
                   'Winner': 'lightgreen', 'Loser': 'salmon'}, legend=False) # Corrected: Added hue and legend=False
    plt.title(f'{stat} Distribution (Violin): Winner vs. Loser')
    plt.ylabel(stat)
    plt.xlabel('Combat Outcome')
    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f'pokemon_{stat.replace(" ", "_")}_violin_winner_vs_loser.png'))
    plt.close()

print(f"\\nAnalysis complete. All outputs saved to '{output_dir}' directory.")
