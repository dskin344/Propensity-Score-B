import sys
import tyro
import pandas as pd
import numpy as np


from dataclasses import dataclass, field
from statsmodels.stats.multitest import multipletests

from propensity_score_matching.utils import load_two_sheet_data, extract_complications, estimate_risk_ratio

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
    
    covariate_cols: list = field(default_factory=lambda: ["age", "BMI", "raceethnic", "diabetes", "HTN", "ICG angiography",
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

    cols = cfg.complications_cols + cfg.covariate_cols
    for col in cols:
        if col not in df_immediate.columns:
            print(f"{xls.sheet_names[0]} missing {col}")
            col_missing = True

    #Separate for loops for readability
    for col in cols:
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
    
    print("\n✓ All required columns are present in both sheets\n")
    
    complications_total, df_total = extract_complications(df_total, cfg.complications_cols)

    print("Total complications found:", complications_total)


    # =============================================================================
    # RUN ACROSS ALL COMPLICATIONS
    # =============================================================================

    df_total = pd.get_dummies(df_total, columns=cfg.categorical_cols, drop_first=True)
    df_total = df_total.astype({col: int for col in df_total.select_dtypes("bool").columns})
    new_cat_cols = [col for col in df_total.columns 
                  if any(col.startswith(category) for category in cfg.categorical_cols)]
    covariate_cols = new_cat_cols + cfg.continuous_cols

    complications_total.update("reoperation")
    results = []
    for complication in complications_total:
        res = estimate_risk_ratio(df_total, complication, "treatment", covariate_cols)
        results.append(res)

    results_df = pd.DataFrame(results)

    # =============================================================================
    # FDR CORRECTION (Benjamini-Hochberg)
    # =============================================================================

    reject, p_fdr, _, _ = multipletests(results_df["p_value"], alpha=cfg.alpha, method="fdr_bh")

    results_df["p_value_fdr"] = p_fdr
    results_df["significant"]  = reject

    # =============================================================================
    # OUTPUT
    # =============================================================================

    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n=== Risk Ratio Results (FDR-corrected) ===\n")
    print(results_df.to_string(index=False))

    # results_df.to_excel("risk_ratio_results.xslx", index=False)
    # print("\nResults saved to risk_ratio_results.xsls")


if __name__ == "__main__":
    main(tyro.cli(Config))
