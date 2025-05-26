"""
Heatmaps for Pokemon Dataset Analysis
- Missing data heatmaps
- Correlation heatmaps
- Type distribution heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Create output directory
os.makedirs('heatmap_analysis', exist_ok=True)


def create_missing_data_heatmaps():
    """Create heatmaps for missing data analysis"""

    # Load datasets
    pokemon_df = pd.read_csv('pokemon.csv')
    combats_df = pd.read_csv('combats.csv')

    print("Creating Missing Data Heatmaps...")

    # Pokemon missing data heatmap
    plt.figure(figsize=(15, 10))

    # Main missing data heatmap
    plt.subplot(2, 2, 1)
    missing_matrix = pokemon_df.isnull()
    sns.heatmap(missing_matrix,
                cbar=True,
                yticklabels=False,
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Missing Data'})
    plt.title('Pokemon Dataset - Missing Data Pattern',
              fontsize=14, fontweight='bold')
    plt.xlabel('Columns')

    # Missing data by column
    plt.subplot(2, 2, 2)
    missing_counts = pokemon_df.isnull().sum()
    missing_counts[missing_counts > 0].plot(kind='bar', color='red', alpha=0.7)
    plt.title('Missing Data Count by Column', fontsize=12, fontweight='bold')
    plt.xlabel('Columns')
    plt.ylabel('Missing Count')
    plt.xticks(rotation=45)

    # Missing data percentage
    plt.subplot(2, 2, 3)
    missing_percentage = (pokemon_df.isnull().sum() / len(pokemon_df)) * 100
    missing_percentage[missing_percentage > 0].plot(
        kind='bar', color='orange', alpha=0.7)
    plt.title('Missing Data Percentage by Column',
              fontsize=12, fontweight='bold')
    plt.xlabel('Columns')
    plt.ylabel('Missing Percentage (%)')
    plt.xticks(rotation=45)

    # Detailed view of Type 2 missing pattern
    plt.subplot(2, 2, 4)
    type2_missing = pokemon_df['Type 2'].isnull()
    type1_counts = pokemon_df.groupby(
        'Type 1')['Type 2'].apply(lambda x: x.isnull().sum())
    type1_counts.plot(kind='bar', color='purple', alpha=0.7)
    plt.title('Type 2 Missing Data by Type 1', fontsize=12, fontweight='bold')
    plt.xlabel('Type 1')
    plt.ylabel('Missing Type 2 Count')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('heatmap_analysis/missing_data_heatmaps.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Combats missing data (should be minimal)
    plt.figure(figsize=(10, 6))
    combats_missing = combats_df.isnull()
    if combats_missing.sum().sum() > 0:
        sns.heatmap(combats_missing, cbar=True,
                    yticklabels=False, cmap='RdYlBu_r')
        plt.title('Combats Dataset - Missing Data Pattern',
                  fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No Missing Data in Combats Dataset',
                 ha='center', va='center', fontsize=16, fontweight='bold')
        plt.title('Combats Dataset - Missing Data Check',
                  fontsize=14, fontweight='bold')

    plt.savefig('heatmap_analysis/combats_missing_data.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_correlation_heatmaps():
    """Create correlation heatmaps for numeric variables"""

    pokemon_df = pd.read_csv('pokemon.csv')

    print("Creating Correlation Heatmaps...")

    # Select numeric columns for correlation
    numeric_cols = ['HP', 'Attack', 'Defense',
                    'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']

    # Main correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = pokemon_df[numeric_cols].corr()

    # Full correlation heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Pokemon Stats Correlation Matrix',
              fontsize=14, fontweight='bold')

    # Strong correlations only (|r| > 0.5)
    plt.subplot(2, 2, 2)
    strong_corr = correlation_matrix.copy()
    strong_corr[abs(strong_corr) < 0.5] = 0
    sns.heatmap(strong_corr,
                annot=True,
                cmap='coolwarm',
                center=0,
                square=True,
                fmt='.2f')
    plt.title('Strong Correlations (|r| > 0.5)',
              fontsize=12, fontweight='bold')

    # Correlation with Generation
    plt.subplot(2, 2, 3)
    gen_corr = pokemon_df[numeric_cols].corrwith(
        pokemon_df['Generation']).sort_values(ascending=False)
    gen_corr.plot(kind='bar', color='green', alpha=0.7)
    plt.title('Correlation with Generation', fontsize=12, fontweight='bold')
    plt.xlabel('Stats')
    plt.ylabel('Correlation with Generation')
    plt.xticks(rotation=45)

    # Battle stats correlation (HP, Attack, Defense, Sp.Atk, Sp.Def, Speed)
    plt.subplot(2, 2, 4)
    battle_stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    battle_corr = pokemon_df[battle_stats].corr()
    sns.heatmap(battle_corr,
                annot=True,
                cmap='viridis',
                square=True,
                fmt='.2f')
    plt.title('Battle Stats Correlation', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('heatmap_analysis/correlation_heatmaps.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_type_distribution_heatmaps():
    """Create heatmaps for Pokemon type distributions"""

    pokemon_df = pd.read_csv('pokemon.csv')

    print("Creating Type Distribution Heatmaps...")

    # Type combination heatmap
    plt.figure(figsize=(15, 12))

    # Create Type 1 vs Type 2 matrix
    type_combinations = pokemon_df.groupby(
        ['Type 1', 'Type 2']).size().unstack(fill_value=0)

    plt.subplot(2, 2, 1)
    sns.heatmap(type_combinations,
                annot=True,
                fmt='d',
                cmap='Blues',
                cbar_kws={'label': 'Count'})
    plt.title('Type 1 vs Type 2 Combinations', fontsize=14, fontweight='bold')
    plt.xlabel('Type 2')
    plt.ylabel('Type 1')

    # Generation vs Type 1 distribution
    plt.subplot(2, 2, 2)
    gen_type = pokemon_df.groupby(
        ['Generation', 'Type 1']).size().unstack(fill_value=0)
    sns.heatmap(gen_type,
                annot=True,
                fmt='d',
                cmap='Oranges',
                cbar_kws={'label': 'Count'})
    plt.title('Generation vs Type 1 Distribution',
              fontsize=12, fontweight='bold')
    plt.xlabel('Type 1')
    plt.ylabel('Generation')

    # Legendary vs Type distribution
    plt.subplot(2, 2, 3)
    legendary_type = pokemon_df.groupby(
        ['Legendary', 'Type 1']).size().unstack(fill_value=0)
    sns.heatmap(legendary_type,
                annot=True,
                fmt='d',
                cmap='Reds',
                cbar_kws={'label': 'Count'})
    plt.title('Legendary vs Type 1 Distribution',
              fontsize=12, fontweight='bold')
    plt.xlabel('Type 1')
    plt.ylabel('Legendary')

    # Average stats by Type 1
    plt.subplot(2, 2, 4)
    stats_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    type_stats = pokemon_df.groupby('Type 1')[stats_cols].mean()
    sns.heatmap(type_stats,
                annot=True,
                fmt='.1f',
                cmap='plasma',
                cbar_kws={'label': 'Average Value'})
    plt.title('Average Stats by Type 1', fontsize=12, fontweight='bold')
    plt.xlabel('Stats')
    plt.ylabel('Type 1')

    plt.tight_layout()
    plt.savefig('heatmap_analysis/type_distribution_heatmaps.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_advanced_missing_analysis():
    """Create advanced missing data analysis"""

    pokemon_df = pd.read_csv('pokemon.csv')

    print("Creating Advanced Missing Data Analysis...")

    plt.figure(figsize=(15, 10))

    # Missing data pattern by generation
    plt.subplot(2, 3, 1)
    gen_missing = pokemon_df.groupby(
        'Generation')['Type 2'].apply(lambda x: x.isnull().sum())
    gen_total = pokemon_df.groupby('Generation').size()
    gen_missing_pct = (gen_missing / gen_total * 100)
    gen_missing_pct.plot(kind='bar', color='red', alpha=0.7)
    plt.title('Type 2 Missing % by Generation', fontsize=12, fontweight='bold')
    plt.xlabel('Generation')
    plt.ylabel('Missing Percentage')
    plt.xticks(rotation=0)

    # Missing data pattern by legendary status
    plt.subplot(2, 3, 2)
    leg_missing = pokemon_df.groupby(
        'Legendary')['Type 2'].apply(lambda x: x.isnull().sum())
    leg_total = pokemon_df.groupby('Legendary').size()
    leg_missing_pct = (leg_missing / leg_total * 100)
    leg_missing_pct.plot(kind='bar', color='orange', alpha=0.7)
    plt.title('Type 2 Missing % by Legendary Status',
              fontsize=12, fontweight='bold')
    plt.xlabel('Legendary')
    plt.ylabel('Missing Percentage')
    plt.xticks(rotation=0)

    # Missing data correlation with stats
    plt.subplot(2, 3, 3)
    pokemon_df['Type2_Missing'] = pokemon_df['Type 2'].isnull().astype(int)
    stats_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    missing_corr = pokemon_df[stats_cols +
                              ['Type2_Missing']].corr()['Type2_Missing'][:-1]
    missing_corr.plot(kind='bar', color='purple', alpha=0.7)
    plt.title('Stats Correlation with Type 2 Missing',
              fontsize=12, fontweight='bold')
    plt.xlabel('Stats')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)

    # Missingness pattern visualization
    plt.subplot(2, 3, 4)
    missing_pattern = pokemon_df.isnull().astype(int)
    pattern_counts = missing_pattern.groupby(
        missing_pattern.columns.tolist()).size().sort_values(ascending=False)
    if len(pattern_counts) > 1:
        pattern_counts.head(5).plot(kind='bar', color='green', alpha=0.7)
        plt.title('Top Missing Patterns', fontsize=12, fontweight='bold')
        plt.xlabel('Pattern Index')
        plt.ylabel('Count')
    else:
        plt.text(0.5, 0.5, 'Only one missing pattern\n(Type 2 only)',
                 ha='center', va='center', fontsize=12)
        plt.title('Missing Patterns', fontsize=12, fontweight='bold')

    # Stats distribution: Pokemon with vs without Type 2
    plt.subplot(2, 3, 5)
    has_type2 = pokemon_df['Type 2'].notna()
    stats_comparison = pd.DataFrame({
        'With Type 2': pokemon_df[has_type2][stats_cols].mean(),
        'Without Type 2': pokemon_df[~has_type2][stats_cols].mean()
    })
    sns.heatmap(stats_comparison.T, annot=True, fmt='.1f', cmap='RdBu_r')
    plt.title('Avg Stats: With vs Without Type 2',
              fontsize=12, fontweight='bold')

    # Missing data summary table
    plt.subplot(2, 3, 6)
    missing_summary = pd.DataFrame({
        'Column': pokemon_df.columns,
        'Missing_Count': pokemon_df.isnull().sum(),
        'Missing_Pct': (pokemon_df.isnull().sum() / len(pokemon_df) * 100).round(2)
    })

    # Create text summary
    summary_text = f"""Missing Data Summary:

Total Rows: {len(pokemon_df):,}
Columns with Missing Data: {(missing_summary['Missing_Count'] > 0).sum()}
Most Missing Column: {missing_summary.loc[missing_summary['Missing_Count'].idxmax(), 'Column']}
Missing Count: {missing_summary['Missing_Count'].max():,}
Missing Percentage: {missing_summary['Missing_Pct'].max():.1f}%

Complete Cases: {len(pokemon_df.dropna()):,}
Incomplete Cases: {len(pokemon_df) - len(pokemon_df.dropna()):,}"""

    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    plt.title('Missing Data Summary', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('heatmap_analysis/advanced_missing_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run all heatmap analyses"""
    print("=" * 60)
    print("HEATMAP ANALYSIS FOR POKEMON DATASETS")
    print("=" * 60)
    print()

    print("Creating comprehensive heatmap analyses...")
    print("1. Missing data heatmaps")
    print("2. Correlation heatmaps")
    print("3. Type distribution heatmaps")
    print("4. Advanced missing data analysis")
    print()

    # Run all analyses
    create_missing_data_heatmaps()
    create_correlation_heatmaps()
    create_type_distribution_heatmaps()
    create_advanced_missing_analysis()

    print("=" * 60)
    print("HEATMAP ANALYSIS COMPLETED!")
    print("=" * 60)
    print("\nGenerated files in 'heatmap_analysis' folder:")
    print("- missing_data_heatmaps.png")
    print("- combats_missing_data.png")
    print("- correlation_heatmaps.png")
    print("- type_distribution_heatmaps.png")
    print("- advanced_missing_analysis.png")


if __name__ == "__main__":
    main()
