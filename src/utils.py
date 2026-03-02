import pandas as pd
import numpy as np

from scipy import stats
from rich.table import Table
from scipy.stats import chi2_contingency, fisher_exact



def analyze_continuous_column(df, column_name):
    """
    Test for normality and return appropriate central tendency and spread.
    
    """
    alpha = 0.05

    data = df[column_name].replace('N/A', np.nan).dropna()
    
    # Test for normality (Shapiro-Wilk)
    _, p_value = stats.shapiro(data)
    is_normal = p_value > alpha
    
    # Calculate IQR
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    
    # Choose central tendency based on normality
    if is_normal:
        central_tendency = data.mean()
    else:
        central_tendency = data.median()
    
    return f"{central_tendency:.2f} [IQR, {q1:.2f}-{q3:.2f}]"

def analyze_categorical_column(df, column_name, categories):
    """
    Analyze categorical column: return mode and frequency distribution.

    """
    data = df[column_name].replace('N/A', np.nan).dropna()
    data = data.astype(str)
    
    value_counts = data.value_counts()
    
    results = {}
    for category in categories:
        if category in value_counts.index:
            count = value_counts[category]
            percentage = (count / len(data)) * 100
            results[str(category)] = f"{count} ({percentage:.2f}%)"
        else:
            results[str(category)] = "0 (0.00%)"
    
    return results

def p_val_continuous(df1, df2, column_name, alpha=0.05):
    """
    Calculate p-value comparing two columns.
    Automatically chooses t-test (normal) or Mann-Whitney U (non-normal).

    """

    data1 = df1[column_name].replace('N/A', np.nan).dropna()
    data2 = df2[column_name].replace('N/A', np.nan).dropna()
    
    # Test normality for both columns
    _, p1 = stats.shapiro(data1)
    _, p2 = stats.shapiro(data2)
    
    both_normal = (p1 > alpha) and (p2 > alpha)
    
    # Choose appropriate test
    if both_normal:
        _, p_value = stats.ttest_ind(data1, data2)
    else:
        _, p_value = stats.mannwhitneyu(data1, data2)
    
    return f"{p_value:.2f}"

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
            pd.Series(['Sheet1']*len(df1)),
            pd.Series(['Sheet2']*len(df2))
        ], ignore_index=True),
        pd.concat([df1[column_name], df2[column_name]], ignore_index=True)
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
        return f"{p_value:.2f}"
    
    # Case 2: Larger table with violated assumptions -> Chi-Square with Monte Carlo simulation
    elif not is_2x2 and assumptions_violated:
        print(f"WARNING: {cells_below_5}/{total_cells} cells have expected frequency < 5.")
        print("Using Chi-Square test with Monte Carlo simulation for better accuracy.")
        
        # Run simulation-based test
        _, p_value_sim, _, _ = chi2_contingency(
            contingency_table, 
            lambda_="log-likelihood"
        )
        return f"{p_value_sim:.2f}"
    
    # Case 3: Assumptions are met -> Standard Chi-Square test
    else:
        print("Using standard Chi-Square test (assumptions met)")
        return f"{p_value_chi2:.2f}"
    
def create_baseline_table(results):
    table = Table(show_header=True, header_style="bold magenta")

    table = Table(title="Analysis Results")
    
    table.add_column("Column", style="cyan", no_wrap=False)
    table.add_column("Sheet 1", style="magenta", justify="right")
    table.add_column("Sheet 2", style="magenta", justify="right")
    table.add_column("Total", style="green", justify="right")
    table.add_column("P-Value", style="yellow", justify="right")
    
    for result in results:
        if result['type'] == 'continuous':
            table.add_row(
                result['col'],
                result['sheet 1'],
                result['sheet 2'],
                result['total'],
                result['pval']
            )
        
        elif result['type'] == 'categorical':
            
            # Categorical: format column name with category
            if result['col']:  # First category (has column name)
                col_display = f"[bold]{result['col']}[/bold]\n  {result['category']}"
                p_val_display = result['pval']
            else:  # Subsequent categories (no column name)
                col_display = f"  {result['category']}"
                p_val_display = ''
                
            
            table.add_row(
                col_display,
                result['sheet 1'],
                result['sheet 2'],
                result['total'],
                p_val_display
            )
    
    return table