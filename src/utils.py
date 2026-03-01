import pandas as pd

from scipy import stats
from scipy.stats import chi2_contingency



def analyze_continuous_column(df, column_name):
    """
    Test for normality and return appropriate central tendency and spread.
    
    """
    alpha = 0.05
    data = df[column_name].dropna()
    
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
    data = df[column_name].dropna()
    
    # Get mode (most common value)
    mode = data.mode()[0] if len(data.mode()) > 0 else None
    mode_count = (data == mode).sum()
    mode_percentage = (mode_count / len(data)) * 100
    
    # Get value counts
    value_counts = data.value_counts()
    unique_count = data.nunique()
    
    return {
        'column': column_name,
        'mode': mode,
        'mode_count': mode_count,
        'mode_percentage': mode_percentage,
        'unique_values': unique_count,
        'value_counts': value_counts.to_dict()
    }

def p_val_continuous(df, column1, column2, alpha=0.05):
    """
    Calculate p-value comparing two columns.
    Automatically chooses t-test (normal) or Mann-Whitney U (non-normal).

    """
    data1 = df[column1].dropna()
    data2 = df[column2].dropna()
    
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
    Calculate p-value for categorical column using Chi-Square test.
    """
    # Create contingency table
    contingency_table = pd.crosstab(
        pd.concat([
            pd.Series(['Sheet1']*len(df1), index=df1.index),
            pd.Series(['Sheet2']*len(df2), index=df2.index)
        ]),
        pd.concat([df1[column_name], df2[column_name]])
    )
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return p_value