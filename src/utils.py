import pandas as pd
import numpy as np

from scipy import stats
from rich.table import Table
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler



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

def load_two_sheet_data(input_file):
    """
    Load data from two sheets in Excel file.
    Sheet 1 = Control (treatment=0)
    Sheet 2 = Treatment (treatment=1)
    """
    print("\n" + "="*60)
    print("LOADING DATA FROM TWO SHEETS")
    print("="*60)
    
    # Read Excel file
    xls = pd.ExcelFile(input_file)
    
    if len(xls.sheet_names) < 2:
        raise ValueError(f"Excel file must have at least 2 sheets. Found: {len(xls.sheet_names)}")
    
    # Read first two sheets
    sheet1_name = xls.sheet_names[0]
    sheet2_name = xls.sheet_names[1]
    
    print(f"\nReading sheets:")
    print(f"  Sheet 1 '{sheet1_name}' → Control group (treatment=0)")
    print(f"  Sheet 2 '{sheet2_name}' → Treatment group (treatment=1)")
    
    control_df = pd.read_excel(input_file, sheet_name=sheet1_name)
    treatment_df = pd.read_excel(input_file, sheet_name=sheet2_name)
    
    print(f"\nOriginal sample sizes:")
    print(f"  Control:   {len(control_df)} observations")
    print(f"  Treatment: {len(treatment_df)} observations")
    print(f"  Total:     {len(control_df) + len(treatment_df)} observations")
    
    # Return combined dataframe for now - column validation will happen later with columns_for_matching
    
    # Add treatment indicator
    control_df['treatment'] = 0
    treatment_df['treatment'] = 1
    
    # Combine into single dataframe
    combined_df = pd.concat([control_df, treatment_df], ignore_index=True)
    
    print(f"\nCombined dataset:")
    print(f"  Total rows: {len(combined_df)}")
    print(f"  Total columns: {len(combined_df.columns)}")
    print(f"  Column names: {', '.join(combined_df.columns.tolist())}")
    
    return combined_df, control_df, treatment_df

def calculate_propensity_scores(df, covariates, treatment_col='treatment'):
    """
    Calculate propensity scores using logistic regression.
    """
    print("\n" + "="*60)
    print("STEP 1: CALCULATING PROPENSITY SCORES")
    print("="*60)
    
    # Prepare features
    X = df[covariates].copy()
    y = df[treatment_col]
    
    # Handle categorical variables (one-hot encoding)
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Standardize continuous variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Fit logistic regression
    print(f"\nFitting logistic regression with {len(covariates)} covariates...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    lr_model.fit(X_scaled, y)
    
    # Calculate propensity scores (probability of treatment)
    propensity_scores = lr_model.predict_proba(X_scaled)[:, 1]
    
    print(f"\nPropensity Score Summary:")
    print(f"  Mean: {propensity_scores.mean():.4f}")
    print(f"  Std:  {propensity_scores.std():.4f}")
    print(f"  Min:  {propensity_scores.min():.4f}")
    print(f"  Max:  {propensity_scores.max():.4f}")
    
    # By group
    treated_ps = propensity_scores[y == 1]
    control_ps = propensity_scores[y == 0]
    
    print(f"\nBy Group:")
    print(f"  Treatment (n={len(treated_ps)}): Mean PS = {treated_ps.mean():.4f}")
    print(f"  Control (n={len(control_ps)}):   Mean PS = {control_ps.mean():.4f}")
    
    return propensity_scores

def propensity_score_matching(df, treatment_col='treatment', ps_col='propensity_score', 
                               caliper=0.03):
    """
    Perform 1:1 nearest neighbor propensity score matching with caliper.
    """
    print("\n" + "="*60)
    print("STEP 2: PERFORMING PROPENSITY SCORE MATCHING")
    print("="*60)
    
    treated = df[df[treatment_col] == 1].copy()
    control = df[df[treatment_col] == 0].copy()
    
    print(f"\nOriginal Sample Sizes:")
    print(f"  Treatment: {len(treated)}")
    print(f"  Control:   {len(control)}")
    print(f"  Total:     {len(df)}")
    
    # Nearest neighbor matching
    print(f"\nPerforming 1:1 nearest neighbor matching with caliper = {caliper}...")
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(control[[ps_col]])
    
    distances, indices = nn.kneighbors(treated[[ps_col]])
    
    # Apply caliper
    matched_treated_list = []
    matched_control_list = []
    used_control_indices = set()
    
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        control_idx = control.index[idx]
        
        # Check if within caliper and not already used
        if dist <= caliper and control_idx not in used_control_indices:
            matched_treated_list.append(treated.iloc[i])
            matched_control_list.append(control.loc[control_idx])
            used_control_indices.add(control_idx)
    
    # Create matched dataframes
    matched_treated = pd.DataFrame(matched_treated_list).reset_index(drop=True)
    matched_control = pd.DataFrame(matched_control_list).reset_index(drop=True)
    
    print(f"\nMatched Sample Sizes:")
    print(f"  Treatment: {len(matched_treated)}")
    print(f"  Control:   {len(matched_control)}")
    print(f"  Total:     {len(matched_treated) + len(matched_control)}")
    
    # Calculate matching statistics
    if len(matched_treated) > 0:
        ps_diff = abs(matched_treated[ps_col].values - matched_control[ps_col].values)
        print(f"\nPropensity Score Matching Quality:")
        print(f"  Mean PS difference: {ps_diff.mean():.4f}")
        print(f"  Max PS difference:  {ps_diff.max():.4f}")
        print(f"  % within 0.01:      {(ps_diff < 0.01).mean()*100:.1f}%")
        print(f"  % within 0.02:      {(ps_diff < 0.02).mean()*100:.1f}%")
    
    # Calculate percentage matched
    pct_treated_matched = (len(matched_treated) / len(treated)) * 100
    pct_control_matched = (len(matched_control) / len(control)) * 100
    
    print(f"\nMatching Success Rate:")
    print(f"  Treatment matched: {pct_treated_matched:.1f}%")
    print(f"  Control matched:   {pct_control_matched:.1f}%")
    
    return matched_treated, matched_control

def calculate_standardized_mean_difference(treated, control, var):
    """
    Calculate standardized mean difference (SMD) for balance assessment.
    """
    try:
        treated_numeric = pd.to_numeric(treated[var], errors='coerce')
        control_numeric = pd.to_numeric(control[var], errors='coerce')
        
        if treated[var].dtype in ['float64', 'int64', 'int32', 'float32'] and treated[var].nunique() > 10:
            mean_diff = treated_numeric.mean() - control_numeric.mean()
            pooled_std = np.sqrt((treated_numeric.std()**2 + control_numeric.std()**2) / 2)
            if pooled_std == 0 or pd.isna(pooled_std):
                return 0
            return abs(mean_diff / pooled_std)
    except:
        pass
    
    # For categorical variables
    if treated[var].nunique() == 2:
        p1 = (treated[var] == treated[var].mode()[0]).mean()
        p2 = (control[var] == control[var].mode()[0]).mean()
        return abs(p1 - p2)
    else:
        return np.nan


def assess_balance(matched_treated, matched_control, covariates):
    """
    Assess covariate balance after matching.
    """
    print("\n" + "="*60)
    print("STEP 3: ASSESSING COVARIATE BALANCE")
    print("="*60)
    
    balance_results = []
    
    for var in covariates:
        treated_vals = matched_treated[var]
        control_vals = matched_control[var]
        
        smd = calculate_standardized_mean_difference(matched_treated, matched_control, var)
        
        is_continuous = False
        try:
            if matched_treated[var].dtype in ['float64', 'int64', 'float32', 'int32'] and matched_treated[var].nunique() > 10:
                is_continuous = True
        except:
            pass
        
        if is_continuous:
            try:
                stat, p_value = stats.ttest_ind(treated_vals, control_vals)
                treated_summary = f"{treated_vals.mean():.2f} ± {treated_vals.std():.2f}"
                control_summary = f"{control_vals.mean():.2f} ± {control_vals.std():.2f}"
            except:
                p_value = np.nan
                treated_summary = "Error"
                control_summary = "Error"
        else:
            try:
                contingency = pd.crosstab(
                    pd.Series(['Treatment']*len(treated_vals) + ['Control']*len(control_vals)),
                    pd.concat([treated_vals.reset_index(drop=True), control_vals.reset_index(drop=True)])
                )
                if contingency.shape == (2, 2) and contingency.min().min() < 5:
                    stat, p_value = stats.fisher_exact(contingency)
                else:
                    stat, p_value, _, _ = stats.chi2_contingency(contingency)
                
                treated_mode = treated_vals.mode()[0] if len(treated_vals.mode()) > 0 else treated_vals.iloc[0]
                control_mode = control_vals.mode()[0] if len(control_vals.mode()) > 0 else control_vals.iloc[0]
                treated_pct = (treated_vals == treated_mode).mean() * 100
                control_pct = (control_vals == control_mode).mean() * 100
                treated_summary = f"{treated_pct:.1f}% ({treated_mode})"
                control_summary = f"{control_pct:.1f}% ({control_mode})"
            except Exception as e:
                p_value = np.nan
                treated_summary = "N/A"
                control_summary = "N/A"
        
        if pd.notna(smd):
            if smd < 0.1:
                balance_status = "✓ Excellent"
            elif smd < 0.2:
                balance_status = "○ Acceptable"
            else:
                balance_status = "✗ Poor"
        else:
            balance_status = "N/A"
        
        balance_results.append({
            'Variable': var,
            'Treatment': treated_summary,
            'Control': control_summary,
            'SMD': f"{smd:.3f}" if pd.notna(smd) else "N/A",
            'p-value': f"{p_value:.3f}" if pd.notna(p_value) else "N/A",
            'Balance': balance_status
        })
    
    balance_df = pd.DataFrame(balance_results)
    
    print("\nCovariate Balance After Matching:")
    print(balance_df.to_string(index=False))
    
    smd_values = pd.to_numeric(balance_df['SMD'], errors='coerce').dropna()
    if len(smd_values) > 0:
        excellent = (smd_values < 0.1).sum()
        acceptable = ((smd_values >= 0.1) & (smd_values < 0.2)).sum()
        poor = (smd_values >= 0.2).sum()
        
        print(f"\nBalance Summary:")
        print(f"  Excellent (SMD < 0.1):  {excellent}/{len(smd_values)}")
        print(f"  Acceptable (SMD < 0.2): {acceptable}/{len(smd_values)}")
        print(f"  Poor (SMD ≥ 0.2):       {poor}/{len(smd_values)}")
    
    return balance_df