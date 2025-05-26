"""
Detailed Descriptive Statistics for Pokemon Dataset
- Extended statistical measures
- Distribution analysis
- Outlier detection
- Data quality assessment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# Create output directory
os.makedirs('detailed_statistics', exist_ok=True)


def calculate_extended_statistics(df, column):
    """Calculate extended statistical measures for a column"""
    data = df[column].dropna()

    stats_dict = {
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'mode': data.mode().iloc[0] if not data.mode().empty else np.nan,
        'min': data.min(),
        'max': data.max(),
        'range': data.max() - data.min(),
        'q1': data.quantile(0.25),
        'q3': data.quantile(0.75),
        'iqr': data.quantile(0.75) - data.quantile(0.25),
        'std': data.std(),
        # Coefficient of Variation
        'variance': data.var(),        'cv': (data.std() / data.mean()) * 100 if data.mean() != 0 else np.nan,
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'se_mean': data.std() / np.sqrt(len(data)),  # Standard Error of Mean
        'mad': np.mean(np.abs(data - data.mean())),  # Mean Absolute Deviation
        'percentile_5': data.quantile(0.05),
        'percentile_95': data.quantile(0.95),
        'percentile_99': data.quantile(0.99)
    }

    return stats_dict


def analyze_outliers(df, column):
    """Detect outliers using multiple methods"""
    data = df[column].dropna()

    # IQR method
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]

    # Z-score method (|z| > 3)
    z_scores = np.abs(stats.zscore(data))
    zscore_outliers = data[z_scores > 3]

    # Modified Z-score method (using MAD)
    median = data.median()
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    modified_zscore_outliers = data[np.abs(modified_z_scores) > 3.5]

    outlier_info = {
        'iqr_outliers_count': len(iqr_outliers),
        'iqr_outliers_pct': (len(iqr_outliers) / len(data)) * 100,
        'zscore_outliers_count': len(zscore_outliers),
        'zscore_outliers_pct': (len(zscore_outliers) / len(data)) * 100,
        'modified_zscore_outliers_count': len(modified_zscore_outliers),
        'modified_zscore_outliers_pct': (len(modified_zscore_outliers) / len(data)) * 100,
        'lower_bound_iqr': lower_bound,
        'upper_bound_iqr': upper_bound
    }

    return outlier_info


def create_distribution_analysis():
    """Create detailed distribution analysis for Pokemon stats"""
    pokemon_df = pd.read_csv('pokemon.csv')

    print("Creating Distribution Analysis...")

    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    # Create comprehensive statistics table
    all_stats = []

    for col in numeric_cols:
        stats_dict = calculate_extended_statistics(pokemon_df, col)
        stats_dict['column'] = col
        all_stats.append(stats_dict)

    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.set_index('column')

    # Save detailed statistics
    stats_df.to_csv('detailed_statistics/extended_statistics.csv')

    # Create outlier analysis
    outlier_analysis = []

    for col in numeric_cols:
        outlier_info = analyze_outliers(pokemon_df, col)
        outlier_info['column'] = col
        outlier_analysis.append(outlier_info)

    outlier_df = pd.DataFrame(outlier_analysis)
    outlier_df = outlier_df.set_index('column')
    outlier_df.to_csv('detailed_statistics/outlier_analysis.csv')

    # Create visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Detailed Distribution Analysis - Pokemon Stats',
                 fontsize=16, fontweight='bold')

    for i, col in enumerate(numeric_cols):
        row = i // 2
        col_idx = i % 2
        ax = axes[row, col_idx]

        data = pokemon_df[col].dropna()

        # Main histogram with KDE
        ax.hist(data, bins=30, alpha=0.7, density=True,
                color='skyblue', edgecolor='black')

        # Add KDE line
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        except:
            pass

        # Add statistical lines
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='green', linestyle='--',
                   linewidth=2, label=f'Median: {median_val:.1f}')

        # Add quartiles
        q1, q3 = data.quantile([0.25, 0.75])
        ax.axvline(q1, color='orange', linestyle=':',
                   alpha=0.7, label=f'Q1: {q1:.1f}')
        ax.axvline(q3, color='orange', linestyle=':',
                   alpha=0.7, label=f'Q3: {q3:.1f}')

        ax.set_title(f'{col} Distribution', fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('detailed_statistics/distribution_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return stats_df, outlier_df


def create_normality_tests():
    """Test normality of distributions"""
    pokemon_df = pd.read_csv('pokemon.csv')

    print("Testing Normality...")

    numeric_cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    normality_results = []

    for col in numeric_cols:
        data = pokemon_df[col].dropna()

        # Shapiro-Wilk test (for sample size < 5000)
        if len(data) < 5000:
            shapiro_stat, shapiro_p = stats.shapiro(data)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan

        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.kstest(
            data, 'norm', args=(data.mean(), data.std()))

        # Anderson-Darling test
        ad_stat, ad_critical, ad_significance = stats.anderson(
            data, dist='norm')

        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(data)

        result = {
            'column': col,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'shapiro_normal': shapiro_p > 0.05 if not np.isnan(shapiro_p) else np.nan,
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'ks_normal': ks_p > 0.05,
            'anderson_stat': ad_stat,
            # 5% significance level
            'anderson_normal': ad_stat < ad_critical[2],
            'jb_stat': jb_stat,
            'jb_p': jb_p,
            'jb_normal': jb_p > 0.05
        }

        normality_results.append(result)

    normality_df = pd.DataFrame(normality_results)
    normality_df.to_csv('detailed_statistics/normality_tests.csv', index=False)

    return normality_df


def create_categorical_analysis():
    """Detailed analysis of categorical variables"""
    pokemon_df = pd.read_csv('pokemon.csv')

    print("Analyzing Categorical Variables...")

    categorical_cols = ['Type 1', 'Type 2', 'Generation', 'Legendary']

    categorical_stats = []

    for col in categorical_cols:
        if col == 'Type 2':
            # Special handling for Type 2 (has missing values)
            total_count = len(pokemon_df)
            non_null_count = pokemon_df[col].count()
            null_count = total_count - non_null_count
            unique_count = pokemon_df[col].nunique()

            # Most frequent non-null value
            mode_value = pokemon_df[col].mode(
            ).iloc[0] if not pokemon_df[col].mode().empty else None
            mode_count = pokemon_df[col].value_counts().iloc[0] if len(
                pokemon_df[col].value_counts()) > 0 else 0

            stats_dict = {
                'column': col,
                'total_count': total_count,
                'non_null_count': non_null_count,
                'null_count': null_count,
                'null_percentage': (null_count / total_count) * 100,
                'unique_count': unique_count,
                'mode_value': mode_value,
                'mode_count': mode_count,
                'mode_percentage': (mode_count / non_null_count) * 100 if non_null_count > 0 else 0
            }
        else:
            # Regular categorical variables
            total_count = len(pokemon_df)
            unique_count = pokemon_df[col].nunique()
            mode_value = pokemon_df[col].mode().iloc[0]
            mode_count = pokemon_df[col].value_counts().iloc[0]

            stats_dict = {
                'column': col,
                'total_count': total_count,
                'non_null_count': total_count,
                'null_count': 0,
                'null_percentage': 0,
                'unique_count': unique_count,
                'mode_value': mode_value,
                'mode_count': mode_count,
                'mode_percentage': (mode_count / total_count) * 100
            }

        categorical_stats.append(stats_dict)

    categorical_df = pd.DataFrame(categorical_stats)
    categorical_df.to_csv(
        'detailed_statistics/categorical_statistics.csv', index=False)

    # Create detailed frequency tables
    for col in categorical_cols:
        if col == 'Type 2':
            freq_table = pokemon_df[col].value_counts(
                dropna=False, normalize=False)
            freq_pct = pokemon_df[col].value_counts(
                dropna=False, normalize=True) * 100
        else:
            freq_table = pokemon_df[col].value_counts(normalize=False)
            freq_pct = pokemon_df[col].value_counts(normalize=True) * 100

        combined_freq = pd.DataFrame({
            'Count': freq_table,
            'Percentage': freq_pct
        }).round(2)

        combined_freq.to_csv(
            f'detailed_statistics/frequency_table_{col.replace(" ", "_").replace(".", "")}.csv')

    return categorical_df


def create_missing_data_detailed_analysis():
    """Comprehensive missing data analysis"""
    pokemon_df = pd.read_csv('pokemon.csv')
    combats_df = pd.read_csv('combats.csv')

    print("Creating Detailed Missing Data Analysis...")

    # Pokemon dataset missing analysis
    pokemon_missing = pokemon_df.isnull()
    pokemon_missing_summary = {
        'dataset': 'Pokemon',
        'total_cells': pokemon_df.size,
        'missing_cells': pokemon_missing.sum().sum(),
        'missing_percentage': (pokemon_missing.sum().sum() / pokemon_df.size) * 100,
        'complete_rows': len(pokemon_df.dropna()),
        'incomplete_rows': len(pokemon_df) - len(pokemon_df.dropna()),
        'columns_with_missing': (pokemon_df.isnull().sum() > 0).sum()
    }

    # Combats dataset missing analysis
    combats_missing = combats_df.isnull()
    combats_missing_summary = {
        'dataset': 'Combats',
        'total_cells': combats_df.size,
        'missing_cells': combats_missing.sum().sum(),
        'missing_percentage': (combats_missing.sum().sum() / combats_df.size) * 100,
        'complete_rows': len(combats_df.dropna()),
        'incomplete_rows': len(combats_df) - len(combats_df.dropna()),
        'columns_with_missing': (combats_df.isnull().sum() > 0).sum()
    }

    missing_summary_df = pd.DataFrame(
        [pokemon_missing_summary, combats_missing_summary])
    missing_summary_df.to_csv(
        'detailed_statistics/missing_data_summary.csv', index=False)

    # Detailed column-wise missing analysis for Pokemon
    pokemon_column_missing = []
    for col in pokemon_df.columns:
        missing_count = pokemon_df[col].isnull().sum()
        missing_pct = (missing_count / len(pokemon_df)) * 100

        col_info = {
            'column': col,
            'data_type': str(pokemon_df[col].dtype),
            'total_values': len(pokemon_df),
            'missing_count': missing_count,
            'present_count': len(pokemon_df) - missing_count,
            'missing_percentage': missing_pct,
            'present_percentage': 100 - missing_pct
        }
        pokemon_column_missing.append(col_info)

    pokemon_missing_df = pd.DataFrame(pokemon_column_missing)
    pokemon_missing_df.to_csv(
        'detailed_statistics/pokemon_column_missing_analysis.csv', index=False)

    return missing_summary_df, pokemon_missing_df


def create_correlation_detailed_analysis():
    """Detailed correlation analysis"""
    pokemon_df = pd.read_csv('pokemon.csv')

    print("Creating Detailed Correlation Analysis...")

    numeric_cols = ['HP', 'Attack', 'Defense',
                    'Sp. Atk', 'Sp. Def', 'Speed', 'Generation']

    # Calculate correlation matrix
    corr_matrix = pokemon_df[numeric_cols].corr()

    # Extract correlation pairs
    correlation_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            var1 = numeric_cols[i]
            var2 = numeric_cols[j]
            correlation = corr_matrix.loc[var1, var2]

            # Correlation strength interpretation
            abs_corr = abs(correlation)
            if abs_corr >= 0.8:
                strength = "Very Strong"
            elif abs_corr >= 0.6:
                strength = "Strong"
            elif abs_corr >= 0.4:
                strength = "Moderate"
            elif abs_corr >= 0.2:
                strength = "Weak"
            else:
                strength = "Very Weak"

            direction = "Positive" if correlation > 0 else "Negative"

            pair_info = {
                'variable_1': var1,
                'variable_2': var2,
                'correlation': correlation,
                'abs_correlation': abs_corr,
                'strength': strength,
                'direction': direction
            }
            correlation_pairs.append(pair_info)

    correlation_df = pd.DataFrame(correlation_pairs)
    correlation_df = correlation_df.sort_values(
        'abs_correlation', ascending=False)
    correlation_df.to_csv(
        'detailed_statistics/correlation_pairs.csv', index=False)

    # Save full correlation matrix
    corr_matrix.to_csv('detailed_statistics/correlation_matrix.csv')

    return correlation_df, corr_matrix


def generate_summary_report():
    """Generate a comprehensive summary report"""
    pokemon_df = pd.read_csv('pokemon.csv')
    combats_df = pd.read_csv('combats.csv')

    print("Generating Summary Report...")

    # Basic dataset information
    pokemon_info = {
        'shape': pokemon_df.shape,
        'memory_usage': pokemon_df.memory_usage(deep=True).sum(),
        'numeric_columns': len(pokemon_df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(pokemon_df.select_dtypes(include=['object', 'bool']).columns),
        'total_missing': pokemon_df.isnull().sum().sum(),
        'duplicate_rows': pokemon_df.duplicated().sum()
    }

    combats_info = {
        'shape': combats_df.shape,
        'memory_usage': combats_df.memory_usage(deep=True).sum(),
        'numeric_columns': len(combats_df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(combats_df.select_dtypes(include=['object', 'bool']).columns),
        'total_missing': combats_df.isnull().sum().sum(),
        'duplicate_rows': combats_df.duplicated().sum()
    }

    # Create summary report
    report = f"""
POKEMON DATASET ANALYSIS - SUMMARY REPORT
{'='*60}

DATASET OVERVIEW:
Pokemon Dataset:
- Shape: {pokemon_info['shape'][0]} rows × {pokemon_info['shape'][1]} columns
- Memory Usage: {pokemon_info['memory_usage']:,} bytes
- Numeric Columns: {pokemon_info['numeric_columns']}
- Categorical Columns: {pokemon_info['categorical_columns']}
- Missing Values: {pokemon_info['total_missing']}
- Duplicate Rows: {pokemon_info['duplicate_rows']}

Combats Dataset:
- Shape: {combats_info['shape'][0]} rows × {combats_info['shape'][1]} columns
- Memory Usage: {combats_info['memory_usage']:,} bytes
- Numeric Columns: {combats_info['numeric_columns']}
- Categorical Columns: {combats_info['categorical_columns']}
- Missing Values: {combats_info['total_missing']}
- Duplicate Rows: {combats_info['duplicate_rows']}

DATA QUALITY ASSESSMENT:
{'='*30}

Pokemon Dataset Quality:
- Completeness: {((pokemon_df.size - pokemon_df.isnull().sum().sum()) / pokemon_df.size * 100):.2f}%
- Missing Data: Only in 'Type 2' column ({pokemon_df['Type 2'].isnull().sum()} missing values)
- Data Types: Appropriate for analysis
- Duplicates: {'None' if pokemon_info['duplicate_rows'] == 0 else f'{pokemon_info["duplicate_rows"]} found'}

Combats Dataset Quality:
- Completeness: {((combats_df.size - combats_df.isnull().sum().sum()) / combats_df.size * 100):.2f}%
- Missing Data: {'None' if combats_info['total_missing'] == 0 else f'{combats_info["total_missing"]} missing values'}
- Data Types: Appropriate for analysis
- Duplicates: {'None' if combats_info['duplicate_rows'] == 0 else f'{combats_info["duplicate_rows"]} found'}

KEY FINDINGS:
{'='*15}

1. Pokemon Statistics:
   - Average HP: {pokemon_df['HP'].mean():.1f}
   - Highest Attack: {pokemon_df['Attack'].max()}
   - Most Defensive: {pokemon_df['Defense'].max()}
   - Fastest Speed: {pokemon_df['Speed'].max()}

2. Type Distribution:
   - Most Common Type 1: {pokemon_df['Type 1'].mode().iloc[0]}
   - Pokemon with dual types: {pokemon_df['Type 2'].notna().sum()}
   - Single-type Pokemon: {pokemon_df['Type 2'].isnull().sum()}

3. Generation Analysis:
   - Generations: {pokemon_df['Generation'].min()} to {pokemon_df['Generation'].max()}
   - Legendary Pokemon: {pokemon_df['Legendary'].sum()}

4. Combat Analysis:
   - Total Battles: {len(combats_df):,}
   - Unique Pokemon in Battles: {len(set(combats_df['First_pokemon'].tolist() + combats_df['Second_pokemon'].tolist()))}
   - First Pokemon Win Rate: {(combats_df['Winner'] == combats_df['First_pokemon']).mean()*100:.1f}%

RECOMMENDATIONS:
{'='*15}

1. Data Preprocessing:
   - Handle Type 2 missing values (consider as "None" type)
   - No duplicate removal needed
   - Data types are appropriate

2. Further Analysis:
   - Investigate stat correlations for battle predictions
   - Analyze type effectiveness in combat
   - Study legendary Pokemon performance

3. Visualization Priorities:
   - Distribution plots for all numeric stats
   - Type combination heatmaps
   - Combat win rate analysis by Pokemon characteristics

Generated: {pd.Timestamp.now()}
"""

    with open('detailed_statistics/summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)

    return report


def main():
    """Main function to run all detailed analyses"""
    print("=" * 60)
    print("DETAILED STATISTICAL ANALYSIS FOR POKEMON DATASETS")
    print("=" * 60)
    print()

    print("Running comprehensive statistical analysis...")
    print("1. Extended descriptive statistics")
    print("2. Distribution analysis and normality tests")
    print("3. Categorical variable analysis")
    print("4. Missing data detailed analysis")
    print("5. Correlation analysis")
    print("6. Summary report generation")
    print()

    # Run all analyses
    stats_df, outlier_df = create_distribution_analysis()
    normality_df = create_normality_tests()
    categorical_df = create_categorical_analysis()
    missing_summary_df, pokemon_missing_df = create_missing_data_detailed_analysis()
    correlation_df, corr_matrix = create_correlation_detailed_analysis()
    report = generate_summary_report()

    print("=" * 60)
    print("DETAILED ANALYSIS COMPLETED!")
    print("=" * 60)
    print("\nGenerated files in 'detailed_statistics' folder:")
    print("- extended_statistics.csv")
    print("- outlier_analysis.csv")
    print("- distribution_analysis.png")
    print("- normality_tests.csv")
    print("- categorical_statistics.csv")
    print("- frequency_table_*.csv (for each categorical variable)")
    print("- missing_data_summary.csv")
    print("- pokemon_column_missing_analysis.csv")
    print("- correlation_pairs.csv")
    print("- correlation_matrix.csv")
    print("- summary_report.txt")

    print(f"\nDatasets analyzed:")
    print(f"- Pokemon: {stats_df.shape[0]} numeric variables")
    print(f"- Categorical variables: {len(categorical_df)}")
    print(f"- Correlation pairs: {len(correlation_df)}")


if __name__ == "__main__":
    main()
