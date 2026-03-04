import sys
import tyro
import pandas as pd


from dataclasses import dataclass, field
from rich import print

from propensity_score_matching.utils import analyze_continuous_column, analyze_categorical_column, p_val_categorical, p_val_continuous, create_baseline_table, get_all_categories, load_two_sheet_data, extract_complications

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


def main(cfg: Config):
    # Load data from two sheets
    df_total, df_immediate, df_delayed = load_two_sheet_data(cfg.file)
    
    df_immediate.columns = df_immediate.columns.str.strip()
    df_delayed.columns = df_delayed.columns.str.strip()
    df_total.columns = df_total.columns.str.strip()
    
    # Get original sheet names for error messages
    xls = pd.ExcelFile(cfg.file)
    col_missing = False

    cols = cfg.continuous_cols + cfg.categorical_cols

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
    
    print("\n✓ All required columns are present in both sheets")
    
    # Build results list
    results = []
    
    # Process continuous columns
    for column_name in cfg.continuous_cols:
        sheet1_result = analyze_continuous_column(df_immediate, column_name)
        sheet2_result = analyze_continuous_column(df_delayed, column_name)
        total_result = analyze_continuous_column(df_total, column_name)
        p_value_result = p_val_continuous(df_immediate, df_delayed, column_name)
        
        results.append({
            'col': column_name,
            'sheet 1': sheet1_result,
            'sheet 2': sheet2_result,
            'total': total_result,
            'pval': p_value_result,
            'category': None,
            'type':'continuous'
        })
    
    # Process categorical columns
    for column_name in cfg.categorical_cols:
        # Get all unique categories across all sheets
        all_categories = get_all_categories(df_immediate, df_delayed, column_name)

        sheet1_result = analyze_categorical_column(df_immediate, column_name, all_categories)
        sheet2_result = analyze_categorical_column(df_delayed, column_name, all_categories)
        total_result = analyze_categorical_column(df_total, column_name, all_categories)
        p_value_result = p_val_categorical(df_immediate, df_delayed, column_name)
        

        for i, category in enumerate(sorted(all_categories)):
            if i == 0:
                # First row: just column name and p-value, no data
                results.append({
                    'col': column_name,
                    'sheet 1': '',
                    'sheet 2': '',
                    'total': '',
                    'pval': p_value_result,
                    'category': '',
                    'type': 'categorical'
                })

            results.append({
                'col': '',
                'sheet 1': f"{sheet1_result[str(category).strip()]}",
                'sheet 2': f"{sheet2_result[str(category).strip()]}",
                'total': f"{total_result[str(category).strip()]}",
                'pval': '',
                'category': category,
                'type':'categorical'
            })
    
    
    # Build Rich table
    table = create_baseline_table(results)
    
    print(table)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_excel('data/baseline.xlsx', index=False)
    print("\n[green]Results saved to baseline.xlsx[/green]")



if __name__ == "__main__":
    main(tyro.cli(Config))
