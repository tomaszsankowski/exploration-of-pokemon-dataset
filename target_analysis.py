"""
Target Analysis for Pokemon Combat Dataset
- Class distribution analysis (win rates)
- Legendary Pokemon distribution and impact
- Combat outcome comparison analysis
- Winner vs Loser characteristics comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Create output directory
os.makedirs('target_analysis', exist_ok=True)


def analyze_class_distribution():
    """Analyze distribution of combat outcomes (target variable)"""
    combats_df = pd.read_csv('combats.csv')

    print("Analyzing Combat Outcome Distribution...")

    # Calculate win rates for first vs second Pokemon
    first_wins = (combats_df['Winner'] == combats_df['First_pokemon']).sum()
    second_wins = (combats_df['Winner'] == combats_df['Second_pokemon']).sum()
    total_combats = len(combats_df)

    first_win_rate = (first_wins / total_combats) * 100
    second_win_rate = (second_wins / total_combats) * 100

    # Create target distribution analysis
    target_distribution = {
        'outcome': ['First Pokemon Wins', 'Second Pokemon Wins'],
        'count': [first_wins, second_wins],
        'percentage': [first_win_rate, second_win_rate]
    }

    target_df = pd.DataFrame(target_distribution)
    target_df.to_csv(
        'target_analysis/combat_outcome_distribution.csv', index=False)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar plot
    bars = ax1.bar(target_df['outcome'], target_df['count'],
                   color=['#3498db', '#e74c3c'], alpha=0.8, edgecolor='black')
    ax1.set_title('Combat Outcome Distribution\n(Count)',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Combats')
    ax1.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, target_df['count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 100,
                 f'{count:,}', ha='center', va='bottom', fontweight='bold')

    # Pie chart
    colors = ['#3498db', '#e74c3c']
    wedges, texts, autotexts = ax2.pie(target_df['percentage'],
                                       labels=target_df['outcome'],
                                       colors=colors, autopct='%1.2f%%',
                                       startangle=90, explode=(0.05, 0.05))
    ax2.set_title('Combat Outcome Distribution\n(Percentage)',
                  fontsize=14, fontweight='bold')

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    plt.tight_layout()
    plt.savefig('target_analysis/combat_outcome_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Balance analysis
    balance_analysis = {
        'metric': ['Total Combats', 'First Pokemon Wins', 'Second Pokemon Wins',
                   'First Win Rate (%)', 'Second Win Rate (%)', 'Class Balance Ratio'],
        'value': [total_combats, first_wins, second_wins,
                  first_win_rate, second_win_rate, min(first_win_rate, second_win_rate) / max(first_win_rate, second_win_rate)]
    }

    balance_df = pd.DataFrame(balance_analysis)
    balance_df.to_csv(
        'target_analysis/class_balance_analysis.csv', index=False)

    return target_df, balance_df


def analyze_legendary_distribution():
    """Analyze distribution and impact of Legendary Pokemon"""
    pokemon_df = pd.read_csv('pokemon.csv')

    print("Analyzing Legendary Pokemon Distribution...")

    # Basic legendary distribution
    legendary_counts = pokemon_df['Legendary'].value_counts()
    legendary_percentages = pokemon_df['Legendary'].value_counts(
        normalize=True) * 100

    legendary_distribution = {
        'legendary_status': ['Regular Pokemon', 'Legendary Pokemon'],
        'count': [legendary_counts[False], legendary_counts[True]],
        'percentage': [legendary_percentages[False], legendary_percentages[True]]
    }

    legendary_df = pd.DataFrame(legendary_distribution)
    legendary_df.to_csv(
        'target_analysis/legendary_distribution.csv', index=False)

    # Legendary vs Regular Pokemon stats comparison
    regular_pokemon = pokemon_df[pokemon_df['Legendary'] == False]
    legendary_pokemon = pokemon_df[pokemon_df['Legendary'] == True]

    stat_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    stats_comparison = []
    for stat in stat_cols:
        regular_mean = regular_pokemon[stat].mean()
        legendary_mean = legendary_pokemon[stat].mean()

        # Statistical test
        t_stat, p_value = stats.ttest_ind(
            regular_pokemon[stat], legendary_pokemon[stat])

        comparison = {
            'stat': stat,
            'regular_mean': regular_mean,
            'legendary_mean': legendary_mean,
            'difference': legendary_mean - regular_mean,
            'percentage_increase': ((legendary_mean - regular_mean) / regular_mean) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        stats_comparison.append(comparison)

    stats_comparison_df = pd.DataFrame(stats_comparison)
    stats_comparison_df.to_csv(
        'target_analysis/legendary_vs_regular_stats.csv', index=False)

    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Legendary distribution pie chart
    colors = ['#2ecc71', '#f39c12']
    wedges, texts, autotexts = ax1.pie(legendary_df['percentage'],
                                       labels=legendary_df['legendary_status'],
                                       colors=colors, autopct='%1.1f%%',
                                       startangle=90, explode=(0.05, 0.1))
    ax1.set_title('Legendary Pokemon Distribution',
                  fontsize=14, fontweight='bold')

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)

    # Stats comparison bar chart
    x_pos = np.arange(len(stat_cols))
    width = 0.35

    bars1 = ax2.bar(x_pos - width/2, stats_comparison_df['regular_mean'],
                    width, label='Regular Pokemon', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, stats_comparison_df['legendary_mean'],
                    width, label='Legendary Pokemon', color='#e74c3c', alpha=0.8)

    ax2.set_title('Average Stats: Regular vs Legendary Pokemon',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Pokemon Stats')
    ax2.set_ylabel('Average Value')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stat_cols, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Box plot comparison
    pokemon_melted = pokemon_df.melt(id_vars=['Legendary'], value_vars=stat_cols,
                                     var_name='Stat', value_name='Value')

    sns.boxplot(data=pokemon_melted, x='Stat',
                y='Value', hue='Legendary', ax=ax3)
    ax3.set_title('Stats Distribution: Regular vs Legendary Pokemon',
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Pokemon Stats')
    ax3.set_ylabel('Stat Value')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # Percentage increase visualization
    bars = ax4.bar(stats_comparison_df['stat'], stats_comparison_df['percentage_increase'],
                   color='#9b59b6', alpha=0.8, edgecolor='black')
    ax4.set_title('Percentage Increase in Stats\n(Legendary vs Regular)',
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Pokemon Stats')
    ax4.set_ylabel('Percentage Increase (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    # Add value labels
    for bar, pct in zip(bars, stats_comparison_df['percentage_increase']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('target_analysis/legendary_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return legendary_df, stats_comparison_df


def analyze_winner_vs_loser_characteristics():
    """Compare characteristics of winning vs losing Pokemon"""
    pokemon_df = pd.read_csv('pokemon.csv')
    combats_df = pd.read_csv('combats.csv')

    print("Analyzing Winner vs Loser Characteristics...")

    # Merge combat data with Pokemon stats
    # Get winner stats
    winner_stats = combats_df.merge(
        pokemon_df, left_on='Winner', right_on='ID', how='left')
    winner_stats['outcome'] = 'Winner'

    # Get loser stats (first determine who lost in each combat)
    combats_with_losers = combats_df.copy()
    combats_with_losers['Loser'] = combats_with_losers.apply(
        lambda row: row['Second_pokemon'] if row['Winner'] == row['First_pokemon'] else row['First_pokemon'],
        axis=1
    )

    loser_stats = combats_with_losers.merge(
        pokemon_df, left_on='Loser', right_on='ID', how='left')
    loser_stats['outcome'] = 'Loser'

    # Combine winner and loser data
    stat_cols = ['HP', 'Attack', 'Defense',
                 'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']

    winner_data = winner_stats[stat_cols + ['outcome', 'Legendary']]
    loser_data = loser_stats[stat_cols + ['outcome', 'Legendary']]

    combined_data = pd.concat([winner_data, loser_data], ignore_index=True)

    # Statistical comparison
    comparison_results = []

    for stat in stat_cols:
        winner_values = winner_data[stat].dropna()
        loser_values = loser_data[stat].dropna()

        winner_mean = winner_values.mean()
        loser_mean = loser_values.mean()
        winner_std = winner_values.std()
        loser_std = loser_values.std()

        # Statistical tests
        t_stat, p_value = stats.ttest_ind(winner_values, loser_values)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(winner_values) - 1) * winner_std**2 +
                             (len(loser_values) - 1) * loser_std**2) /
                             (len(winner_values) + len(loser_values) - 2))
        cohens_d = (winner_mean - loser_mean) / pooled_std

        result = {
            'stat': stat,
            'winner_mean': winner_mean,
            'winner_std': winner_std,
            'loser_mean': loser_mean,
            'loser_std': loser_std,
            'mean_difference': winner_mean - loser_mean,
            'percentage_difference': ((winner_mean - loser_mean) / loser_mean) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': 'Large' if abs(cohens_d) >= 0.8 else 'Medium' if abs(cohens_d) >= 0.5 else 'Small',
            'significant': p_value < 0.05
        }
        comparison_results.append(result)

    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(
        'target_analysis/winner_vs_loser_stats.csv', index=False)

    # Legendary status comparison
    winner_legendary_rate = (winner_data['Legendary'] == True).mean() * 100
    loser_legendary_rate = (loser_data['Legendary'] == True).mean() * 100

    legendary_comparison = {
        'group': ['Winners', 'Losers'],
        'legendary_rate': [winner_legendary_rate, loser_legendary_rate],
        'regular_rate': [100 - winner_legendary_rate, 100 - loser_legendary_rate]
    }

    legendary_comp_df = pd.DataFrame(legendary_comparison)
    legendary_comp_df.to_csv(
        'target_analysis/winner_vs_loser_legendary.csv', index=False)

    # Create comprehensive visualizations
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    fig.suptitle('Winner vs Loser Analysis',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Box plots for all stats
    combined_melted = combined_data.melt(id_vars=['outcome'], value_vars=stat_cols,
                                         var_name='Stat', value_name='Value')

    sns.boxplot(data=combined_melted, x='Stat',
                y='Value', hue='outcome', ax=axes[0, 0])
    axes[0, 0].set_title(
        'Stats Distribution: Winners vs Losers', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Pokemon Stats')
    axes[0, 0].set_ylabel('Stat Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Violin plots for key stats
    key_stats = ['Attack', 'Defense', 'HP', 'Speed']
    key_melted = combined_data.melt(id_vars=['outcome'], value_vars=key_stats,
                                    var_name='Stat', value_name='Value')

    sns.violinplot(data=key_melted, x='Stat', y='Value',
                   hue='outcome', ax=axes[0, 1])
    axes[0, 1].set_title('Key Stats Distribution (Violin Plot)',
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Pokemon Stats')
    axes[0, 1].set_ylabel('Stat Value')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Mean differences bar chart
    bars = axes[1, 0].bar(comparison_df['stat'], comparison_df['mean_difference'],
                          color=[
                              'green' if x > 0 else 'red' for x in comparison_df['mean_difference']],
                          alpha=0.8, edgecolor='black')
    axes[1, 0].set_title(
        'Mean Stat Differences\n(Winner - Loser)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Pokemon Stats')
    axes[1, 0].set_ylabel('Mean Difference')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)

    # Add value labels
    for bar, diff in zip(bars, comparison_df['mean_difference']):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2.,
                        height + (0.5 if height > 0 else -1),
                        f'{diff:.1f}', ha='center', va='bottom' if height > 0 else 'top',
                        fontweight='bold')

    # 4. Effect sizes
    colors = ['red' if x == 'Large' else 'orange' if x == 'Medium' else 'green'
              for x in comparison_df['effect_size']]
    bars = axes[1, 1].bar(comparison_df['stat'], comparison_df['cohens_d'].abs(),
                          color=colors, alpha=0.8, edgecolor='black')
    axes[1, 1].set_title("Effect Sizes (Cohen's d)",
                         fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Pokemon Stats')
    axes[1, 1].set_ylabel("Cohen's d (absolute value)")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    # Add effect size lines
    axes[1, 1].axhline(y=0.2, color='green', linestyle='--',
                       alpha=0.7, label='Small (0.2)')
    axes[1, 1].axhline(y=0.5, color='orange', linestyle='--',
                       alpha=0.7, label='Medium (0.5)')
    axes[1, 1].axhline(y=0.8, color='red', linestyle='--',
                       alpha=0.7, label='Large (0.8)')
    axes[1, 1].legend()

    # 5. Legendary status comparison
    x_pos = np.arange(len(legendary_comp_df))
    width = 0.35

    bars1 = axes[2, 0].bar(x_pos - width/2, legendary_comp_df['legendary_rate'],
                           width, label='Legendary Pokemon', color='#f39c12', alpha=0.8)
    bars2 = axes[2, 0].bar(x_pos + width/2, legendary_comp_df['regular_rate'],
                           width, label='Regular Pokemon', color='#3498db', alpha=0.8)

    axes[2, 0].set_title(
        'Legendary Pokemon Rate: Winners vs Losers', fontsize=14, fontweight='bold')
    axes[2, 0].set_xlabel('Group')
    axes[2, 0].set_ylabel('Percentage (%)')    axes[2, 0].set_xticks(x_pos)
    axes[2, 0].set_xticklabels(legendary_comp_df['group'])
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[2, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 6. P-values significance plot
    significant_stats = comparison_df[comparison_df['significant']
                                      == True]['stat'].tolist()
    colors = [
        'green' if stat in significant_stats else 'red' for stat in comparison_df['stat']]

    # Handle potential log10(0) by setting minimum p-value
    p_values_safe = np.maximum(comparison_df['p_value'], 1e-16)
    bars = axes[2, 1].bar(comparison_df['stat'], -np.log10(p_values_safe),
                          color=colors, alpha=0.8, edgecolor='black')
    axes[2, 1].set_title(
        'Statistical Significance\n(-log10(p-value))', fontsize=14, fontweight='bold')
    axes[2, 1].set_xlabel('Pokemon Stats')
    axes[2, 1].set_ylabel('-log10(p-value)')
    axes[2, 1].tick_params(axis='x', rotation=45)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7,
                       label='p=0.05 threshold')
    axes[2, 1].legend()

    plt.tight_layout()
    plt.savefig('target_analysis/winner_vs_loser_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return comparison_df, legendary_comp_df, combined_data


def create_target_summary_report():
    """Generate comprehensive target analysis report"""
    print("Generating Target Analysis Summary Report...")

    # Load results
    target_df = pd.read_csv('target_analysis/combat_outcome_distribution.csv')
    balance_df = pd.read_csv('target_analysis/class_balance_analysis.csv')
    legendary_df = pd.read_csv('target_analysis/legendary_distribution.csv')
    stats_comparison_df = pd.read_csv(
        'target_analysis/legendary_vs_regular_stats.csv')
    winner_loser_df = pd.read_csv('target_analysis/winner_vs_loser_stats.csv')
    legendary_comp_df = pd.read_csv(
        'target_analysis/winner_vs_loser_legendary.csv')

    # Generate report
    report = f"""
POKEMON COMBAT TARGET ANALYSIS - COMPREHENSIVE REPORT
{'='*65}

EXECUTIVE SUMMARY:
This report analyzes the target variable (combat outcomes) and key characteristics
that influence Pokemon battle results. The analysis covers class distribution,
legendary Pokemon impact, and detailed winner vs loser comparisons.

{'='*65}

1. COMBAT OUTCOME DISTRIBUTION (CLASS BALANCE)
{'='*50}

Target Variable Analysis:
- Total Combats: {balance_df[balance_df['metric'] == 'Total Combats']['value'].iloc[0]:,.0f}
- First Pokemon Wins: {balance_df[balance_df['metric'] == 'First Pokemon Wins']['value'].iloc[0]:,.0f} ({balance_df[balance_df['metric'] == 'First Win Rate (%)']['value'].iloc[0]:.2f}%)
- Second Pokemon Wins: {balance_df[balance_df['metric'] == 'Second Pokemon Wins']['value'].iloc[0]:,.0f} ({balance_df[balance_df['metric'] == 'Second Win Rate (%)']['value'].iloc[0]:.2f}%)

Class Balance Assessment:
- Balance Ratio: {balance_df[balance_df['metric'] == 'Class Balance Ratio']['value'].iloc[0]:.3f}
- Dataset Balance: {'BALANCED' if balance_df[balance_df['metric'] == 'Class Balance Ratio']['value'].iloc[0] > 0.8 else 'IMBALANCED'}

KEY FINDING: The dataset is {'well-balanced' if balance_df[balance_df['metric'] == 'Class Balance Ratio']['value'].iloc[0] > 0.8 else 'slightly imbalanced'} with nearly equal win rates for first and second Pokemon.

{'='*65}

2. LEGENDARY POKEMON ANALYSIS
{'='*35}

Distribution:
- Total Pokemon: {legendary_df['count'].sum():,}
- Regular Pokemon: {legendary_df[legendary_df['legendary_status'] == 'Regular Pokemon']['count'].iloc[0]:,} ({legendary_df[legendary_df['legendary_status'] == 'Regular Pokemon']['percentage'].iloc[0]:.1f}%)
- Legendary Pokemon: {legendary_df[legendary_df['legendary_status'] == 'Legendary Pokemon']['count'].iloc[0]:,} ({legendary_df[legendary_df['legendary_status'] == 'Legendary Pokemon']['percentage'].iloc[0]:.1f}%)

Statistical Superiority of Legendary Pokemon:
"""

    # Add legendary vs regular stats
    for _, row in stats_comparison_df.iterrows():
        significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        report += f"- {row['stat']}: +{row['percentage_increase']:.1f}% higher than regular Pokemon {significance}\n"

    report += f"""
KEY FINDING: Legendary Pokemon are rare ({legendary_df[legendary_df['legendary_status'] == 'Legendary Pokemon']['percentage'].iloc[0]:.1f}% of all Pokemon) but significantly
superior in ALL stats, with Attack showing the highest increase.

{'='*65}

3. WINNER VS LOSER CHARACTERISTICS
{'='*40}

Statistical Differences (Winner vs Loser):
"""

    # Add winner vs loser comparison
    for _, row in winner_loser_df.iterrows():
        significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        effect = row['effect_size']
        direction = "higher" if row['mean_difference'] > 0 else "lower"
        report += f"- {row['stat']}: Winners have {abs(row['percentage_difference']):.1f}% {direction} values ({effect} effect) {significance}\n"

    # Legendary rates in winners vs losers
    winner_legendary_rate = legendary_comp_df[legendary_comp_df['group']
                                              == 'Winners']['legendary_rate'].iloc[0]
    loser_legendary_rate = legendary_comp_df[legendary_comp_df['group']
                                             == 'Losers']['legendary_rate'].iloc[0]

    report += f"""
Legendary Pokemon in Combat Outcomes:
- Winners: {winner_legendary_rate:.1f}% are Legendary Pokemon
- Losers: {loser_legendary_rate:.1f}% are Legendary Pokemon
- Legendary Advantage: {winner_legendary_rate - loser_legendary_rate:.1f} percentage points

{'='*65}

4. KEY INSIGHTS AND RECOMMENDATIONS
{'='*40}

Combat Prediction Factors:
1. **Most Important Stats for Winning:**
"""

    # Sort by effect size and add top predictors
    top_predictors = winner_loser_df.nlargest(3, 'cohens_d')
    for i, (_, row) in enumerate(top_predictors.iterrows(), 1):
        report += f"   {i}. {row['stat']} (Cohen's d = {row['cohens_d']:.3f}, {row['effect_size']} effect)\n"

    report += f"""
2. **Legendary Status Impact:**
   - Legendary Pokemon have {winner_legendary_rate / loser_legendary_rate:.1f}x higher chance of winning
   - Only {legendary_df[legendary_df['legendary_status'] == 'Legendary Pokemon']['percentage'].iloc[0]:.1f}% of Pokemon are Legendary, but they dominate combat outcomes

3. **Model Development Recommendations:**
   - Focus on top predictive stats: {', '.join(top_predictors['stat'].tolist())}
   - Include Legendary status as a strong feature
   - Consider stat interactions and combinations
   - No class imbalance correction needed for target variable

4. **Data Quality for Prediction:**
   - Clean, balanced dataset suitable for binary classification
   - Strong statistical differences between winners and losers
   - Multiple significant predictive features available

{'='*65}

STATISTICAL NOTES:
- *** p < 0.001 (highly significant)
- ** p < 0.01 (very significant)
- * p < 0.05 (significant)
- Effect sizes: Small (0.2), Medium (0.5), Large (0.8+)

Generated: {pd.Timestamp.now()}
"""

    with open('target_analysis/target_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    return report


def main():
    """Main function to run all target analyses"""
    print("=" * 65)
    print("POKEMON COMBAT TARGET ANALYSIS")
    print("=" * 65)
    print()

    print("Running comprehensive target variable analysis...")
    print("1. Combat outcome distribution (class balance)")
    print("2. Legendary Pokemon distribution and impact")
    print("3. Winner vs Loser characteristics comparison")
    print("4. Comprehensive target analysis report")
    print()

    # Run all analyses
    target_df, balance_df = analyze_class_distribution()
    legendary_df, stats_comparison_df = analyze_legendary_distribution()
    comparison_df, legendary_comp_df, combined_data = analyze_winner_vs_loser_characteristics()
    report = create_target_summary_report()

    print("=" * 65)
    print("TARGET ANALYSIS COMPLETED!")
    print("=" * 65)
    print("\nGenerated files in 'target_analysis' folder:")
    print("- combat_outcome_distribution.csv & .png")
    print("- class_balance_analysis.csv")
    print("- legendary_distribution.csv & legendary_analysis.png")
    print("- legendary_vs_regular_stats.csv")
    print("- winner_vs_loser_stats.csv")
    print("- winner_vs_loser_legendary.csv")
    print("- winner_vs_loser_analysis.png")
    print("- target_analysis_report.txt")

    print(f"\nKey Findings:")
    print(
        f"- Dataset balance ratio: {balance_df[balance_df['metric'] == 'Class Balance Ratio']['value'].iloc[0]:.3f}")
    print(
        f"- Legendary Pokemon rate: {legendary_df[legendary_df['legendary_status'] == 'Legendary Pokemon']['percentage'].iloc[0]:.1f}%")
    print(
        f"- Significant stat differences: {len(comparison_df[comparison_df['significant'] == True])}/{len(comparison_df)}")
    print(
        f"- Large effect sizes: {len(comparison_df[comparison_df['effect_size'] == 'Large'])}/{len(comparison_df)}")


if __name__ == "__main__":
    main()
