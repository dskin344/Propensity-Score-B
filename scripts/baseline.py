import tyro
import pandas as pd


from dataclasses import dataclass, field
from rich import print

from utils import analyze_continuous_column, analyze_categorical_column, p_val_categorical, p_val_continuous, create_baseline_table

@dataclass
class Config():
    file: str
    continuous_cols: list = field(default_factory=lambda: ["age", "BMI"])
    categorical_cols: list = field(default_factory=lambda: ["raceethnic", "diabetes", "HTN", "SPY",
                     "tobacco_history", "alcohol_history", "pre-pec", "sub-pec",
                     "NSM", "SSM", "neoadjuvant chemotherapy (yes=1)",
                     "adjuvant chemotherapy (yes=1)", "immunotherapy (keytruda?)", 
                     "RT (yes=1)", "adjuvant endocrine", "ADM/dermal sling",
                     "SLNB (yes=1)", "ALND (yes=1)", "ER +", "PR+", "HER2+", "grade1", 
                     "mastectomy laterality", "cancer laterality R(0), L (1), both (2)",
                     "clinical stage", "cancer type"])

def main(cfg: Config):
    df_immediate = pd.read_excel(cfg.file, sheet_name="single stage")
    df_delayed = pd.read_excel(cfg.file, sheet_name="two-stage")
    df_total = pd.concat([df_immediate, df_delayed], ignore_index=True)
    
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
        all_categories = set(pd.concat([df_immediate[column_name], df_delayed[column_name]]).dropna().unique())
        all_categories = {str(cat) for cat in all_categories}

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
                'sheet 1': f"{sheet1_result[str(category)]}",
                'sheet 2': f"{sheet2_result[str(category)]}",
                'total': f"{total_result[str(category)]}",
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
    print("\n[green]Results saved to results.csv[/green]")



if __name__ == "__main__":
    main(tyro.cli(Config))
