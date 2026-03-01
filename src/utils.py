import pandas as pd

from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact



def analyze_continuous_column(df, column_name):
    """
    Test for normality and return appropriate central tendency and spread.
    
    """
    alpha = 0.05
    data = df[column_name]
    data = df[data != 'N/A'].dropna()
    
    # Test for normality (Shapiro-Wilk)
    _, p_value = stats.shapiro(data)
    is_normal = p_value > alpha
    
    # Calculate IQR
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    
    # Choose central tendency based on normality
    if is_normal:
        central_tendency = data.mean()
        measure = "Mean"
    else:
        central_tendency = data.median()
        measure = "Median"
    
    return f"{central_tendency} [IQR, {q1}-{q3}]"

def analyze_categorical_column(df, column_name):
    """
    Analyze categorical column: return mode and frequency distribution.

    """
    data = df[column_name]
    data = df[data != 'N/A'].dropna()
    
    value_counts = data.value_counts()
    
    results = {}
    for category, count in value_counts.items():
        percentage = (count / len(data)) * 100
        results[category] = f"{count} ({percentage}%)"
    
    return results

def p_val_continuous(df1, df2, column_name, alpha=0.05):
    """
    Calculate p-value comparing two columns.
    Automatically chooses t-test (normal) or Mann-Whitney U (non-normal).

    """
    data1 = df1[column_name]
    data1 = df1[data1 != 'N/A'].dropna()

    data2 = df2[column_name]
    data2 = df2[data2 != 'N/A'].dropna()
    
    # Test normality for both columns
    _, p1 = stats.shapiro(data1)
    _, p2 = stats.shapiro(data2)
    
    both_normal = (p1 > alpha) and (p2 > alpha)
    
    # Choose appropriate test
    if both_normal:
        _, p_value = stats.ttest_ind(data1, data2)
        test_name = "t-test"
    else:
        _, p_value = stats.mannwhitneyu(data1, data2)
        test_name = "Mann-Whitney U"
    
    return p_value


def p_val_categorical(df1, df2, column_name):
    """
    Calculate p-value for categorical column using the most appropriate test.
    - Uses Fisher's Exact Test for 2x2 tables with small samples
    - Uses Chi-Square test for larger tables when assumptions are met
    - Uses Chi-Square with simulation for larger tables with small samples

    """
    # Create contingency table
    contingency_table = pd.crosstab(
        pd.concat([
            pd.Series(['Sheet1']*len(df1), index=df1.index),
            pd.Series(['Sheet2']*len(df2), index=df2.index)
        ]),
        pd.concat([df1[column_name], df2[column_name]])
    )
    
    # Get expected frequencies for assumption checking
    chi2, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
    
    # Check assumptions
    total_cells = expected.size
    cells_below_5 = (expected < 5).sum()
    cells_below_1 = (expected < 1).sum()
    
    # Determine which test to use
    is_2x2 = contingency_table.shape == (2, 2)
    assumptions_violated = (cells_below_1 > 0) or (cells_below_5 > total_cells * 0.2)
    
    # Case 1: 2x2 table with violated assumptions -> Fisher's Exact Test
    if is_2x2 and assumptions_violated:
        print("Using Fisher's Exact Test (2x2 table with small sample size)")
        _, p_value = fisher_exact(contingency_table)
        return p_value
    
    # Case 2: Larger table with violated assumptions -> Chi-Square with Monte Carlo simulation
    elif not is_2x2 and assumptions_violated:
        print(f"WARNING: {cells_below_5}/{total_cells} cells have expected frequency < 5.")
        print("Using Chi-Square test with Monte Carlo simulation for better accuracy.")
        
        # Run simulation-based test
        chi2_sim, p_value_sim, dof_sim, expected_sim = chi2_contingency(
            contingency_table, 
            lambda_="log-likelihood"
        )
        return p_value_sim
    
    # Case 3: Assumptions are met -> Standard Chi-Square test
    else:
        print("Using standard Chi-Square test (assumptions met)")
        return p_value_chi2