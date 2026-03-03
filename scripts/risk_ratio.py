import sys
import tyro
import pandas as pd
import numpy as np
import statsmodels.api as sm

from dataclasses import dataclass, field
from statsmodels.stats.multitest import multipletests

from  propensity_score_matching.utils import load_two_sheet_data

@dataclass
class Config():
    file: str
    continuous_cols: list = field(default_factory=lambda: ["age", "BMI"])
    categorical_cols: list = field(default_factory=lambda: ["raceethnic", "diabetes", "HTN", "ICG angiography",
                     "tobacco_history", "alcohol_history", "pre-pec", "sub-pec",
                     "NSM", "SSM", "neoadjuvant chemotherapy (yes=1)",
                     "adjuvant chemotherapy (yes=1)", "immunotherapy (keytruda?)", 
                     "RT (yes=1)", "adjuvant endocrine", "ADM/dermal sling",
                     "SLNB (yes=1)", "ALND (yes=1)", "ER +", "PR+", "HER2+", "grade1", 
                     "mastectomy laterality", "cancer laterality R(0), L (1), both (2)",
                     "clinical stage", "cancer type"])
    complications_cols: list = field(default_factory=lambda: ["complications_2", "complications_3", "complications_4"])
    alpha: float = 0.05


# =============================================================================
# LOAD DATA
# =============================================================================
def main(cfg: Config):

    # Load data from two sheets
    df_total, df_immediate, df_delayed = load_two_sheet_data(cfg.file)
    
    df_immediate.columns = df_immediate.columns.str.strip()
    df_delayed.columns = df_delayed.columns.str.strip()
    df_total.columns = df_total.columns.str.strip()
    
    # Get original sheet names for error messages
    xls = pd.ExcelFile(cfg.file)
    col_missing = False

    for col in cfg.cols:
        if col not in df_immediate.columns:
            print(f"{xls.sheet_names[0]} missing {col}")
            col_missing = True

    #Separate for loops for readability
    for col in cfg.cols:
        if col not in df_delayed.columns:
            print(f"{xls.sheet_names[1]} missing {col}")
            col_missing = True
        
    # Exit if any columns are missing
    if col_missing:
        print("\n" + "="*60)
        print("ERROR: MISSING REQUIRED COLUMNS")
        print("="*60)
        print("\nBoth sheets must contain all columns specified for matching.")
        print("1. Add missing columns to your Excel file")
        print("2. Re-run and select only columns that exist in both sheets")
        sys.exit(1)
    
    print("\n✓ All required columns are present in both sheets")

    # =============================================================================
    # RISK RATIO ESTIMATION
    # Poisson regression with robust standard errors directly estimates risk ratios
    # (logistic regression gives odds ratios, which overestimate RR when events are common)
    # =============================================================================

    def estimate_risk_ratio(df, outcome_col, group_col, covariate_cols):
        """
        Fit a Poisson GLM with robust SEs to estimate the risk ratio for the
        treatment group vs. control, adjusted for covariates.
        Returns a dict with: RR, 95% CI, p-value, and N.
        """
        cols = [outcome_col, group_col] + covariate_cols
        subset = df[cols].dropna()

        y = subset[outcome_col]
        X = subset[[group_col] + covariate_cols]
        X = sm.add_constant(X)

        model = sm.GLM(y, X, family=sm.families.Poisson())
        result = model.fit(cov_type="HC0")  # HC0 = robust standard errors

        coef = result.params[group_col]
        se   = result.bse[group_col]
        p    = result.pvalues[group_col]

        return {
            "complication": outcome_col,
            "n":            len(subset),
            "n_events":     int(y.sum()),
            "risk_ratio":   round(np.exp(coef), 3),
            "ci_lower":     round(np.exp(coef - 1.96 * se), 3),
            "ci_upper":     round(np.exp(coef + 1.96 * se), 3),
            "p_value":      p,
        }

    # =============================================================================
    # RUN ACROSS ALL COMPLICATIONS
    # =============================================================================

    results = []
    for complication in COMPLICATION_COLS:
        res = estimate_risk_ratio(df, complication, "group_binary", COVARIATE_COLS)
        results.append(res)

    results_df = pd.DataFrame(results)

    # =============================================================================
    # FDR CORRECTION (Benjamini-Hochberg)
    # =============================================================================

    reject, p_fdr, _, _ = multipletests(results_df["p_value"], alpha=ALPHA, method="fdr_bh")

    results_df["p_value_fdr"] = p_fdr
    results_df["significant"]  = reject

    # =============================================================================
    # OUTPUT
    # =============================================================================

    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n=== Risk Ratio Results (FDR-corrected) ===\n")
    print(results_df.to_string(index=False))

    results_df.to_csv("risk_ratio_results.csv", index=False)
    print("\nResults saved to risk_ratio_results.csv")


if __name__ == "__main__":
    main(tyro.cli(Config))
