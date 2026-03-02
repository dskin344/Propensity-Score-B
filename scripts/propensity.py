import sys
import tyro
import pandas as pd
import numpy as np


from dataclasses import dataclass, field
from rich import print

from propensity_score_matching.utils import load_two_sheet_data, calculate_propensity_scores, propensity_score_matching, assess_balance


@dataclass
class Config():
    file: str
    cols: list = field(default_factory=lambda: ["age", "BMI", "raceethnic", "diabetes", "HTN", "SPY",
                                            "tobacco_history", "alcohol_history", "pre-pec", "sub-pec",
                                            "NSM", "SSM", "neoadjuvant chemotherapy (yes=1)",
                                            "adjuvant chemotherapy (yes=1)", "immunotherapy (keytruda?)", 
                                            "RT (yes=1)", "adjuvant endocrine", "ADM/dermal sling",
                                            "SLNB (yes=1)", "ALND (yes=1)", "ER +", "PR+", "HER2+", "grade1", 
                                            "mastectomy laterality", "cancer laterality R(0), L (1), both (2)",
                                            "clinical stage", "cancer type"])
    caliper: float = 0.3

def main(cfg: Config):
    """
    Main function to run propensity score matching pipeline.
    """
    print("="*60)
    print("PROPENSITY SCORE MATCHING - TWO SHEET INPUT")
    print("="*60)
    
    # Load data from two sheets
    df_combined, df_control, df_treatment = load_two_sheet_data(cfg.file)

    # Validate that both sheets have all required columns
    print("\n" + "-"*60)
    print("VALIDATING COLUMNS IN BOTH SHEETS")
    print("-"*60)
    
    # Get original sheet names for error messages
    xls = pd.ExcelFile(cfg.file)

    for col in cfg.cols:
        if col not in df_control.columns:
            print(f"{xls.sheet_names[0]} missing {col}")
            col_missing = True

    #Separate for loops for readability
    for col in cfg.cols:
        if col not in df_treatment.columns:
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
    
    # Calculate propensity scores
    df_combined['propensity_score'] = calculate_propensity_scores(df_combined, cfg.cols)
    
    # Perform matching
    matched_treated, matched_control = propensity_score_matching(
        df_combined, 
        treatment_col='treatment',
        ps_col='propensity_score',
        caliper=cfg.caliper
    )
    
    if len(matched_treated) == 0:
        print("\n⚠ WARNING: No matches found! Try increasing the caliper.")
        sys.exit(1)
    
    # Assess balance
    balance_df = assess_balance(matched_treated, matched_control, cfg.cols)
    
    # Save results
    print("\n" + "="*60)
    print("STEP 4: SAVING RESULTS TO EXCEL")
    print("="*60)
    
    # Remove treatment column from output (it's redundant - all 1s or all 0s)
    matched_treated_out = matched_treated.drop('treatment', axis=1, errors='ignore')
    matched_control_out = matched_control.drop('treatment', axis=1, errors='ignore')
    
    output_path = f"\data\{cfg.file}_propensity.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Treatment group (matched)
        matched_treated_out.to_excel(writer, sheet_name='Treatment (Matched)', index=False)
        
        # Sheet 2: Control group (matched pairs in same order)
        matched_control_out.to_excel(writer, sheet_name='Control (Matched)', index=False)
        
        # Sheet 3: Balance assessment
        if balance_df is not None:
            balance_df.to_excel(writer, sheet_name='Balance Assessment', index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"\nExcel Structure:\n")
    print(f"  → Row 1 in 'Treatment' matches Row 1 in 'Control'")
    print(f"  → Row 2 in 'Treatment' matches Row 2 in 'Control'")
    print(f"  → And so on...")
    
    print("\n" + "="*60)
    print("MATCHING COMPLETED SUCCESSFULLY!")
    print("="*60)
    


if __name__=="__main__":
    main(tyro.cli(Config))