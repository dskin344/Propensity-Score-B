import tyro
import pandas as pd


from dataclasses import dataclass, field
from rich.table import Table
from rich import print

from utils import analyze_continuous_column, analyze_categorical_column, p_val_categorical, p_val_continuous

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
            'Column': column_name,
            'Sheet 1': sheet1_result,
            'Sheet 2': sheet2_result,
            'Total': total_result,
            'P-Value': p_value_result,
            'Category': None,
            'Type':'continuous'
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
            results.append({
                'Column': column_name if i == 0 else '',
                'Sheet 1': f"{sheet1_result[str(category)]}",
                'Sheet 2': f"{sheet2_result[str(category)]}",
                'Total': f"{total_result[str(category)]}",
                'P-Value': p_value_result if i == 0 else '',
                'Category': category,
                'Type':'categorical'
            })
    
    # Build Rich table
    table = Table(title="Analysis Results")
    table.add_column("Column", style="cyan")
    table.add_column("Category", style="blue")
    table.add_column("Sheet 1", style="magenta")
    table.add_column("Sheet 2", style="magenta")
    table.add_column("Total", style="green")
    table.add_column("P-Value", style="yellow")
    
    for row in results:
        table.add_row(
            str(row['Column']),
            str(row['Category'] if row['Category'] else ''),
            str(row['Sheet 1']),
            str(row['Sheet 2']),
            str(row['Total']),
            str(row['P-Value'])
        )
    
    print(table)
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_excel('data/baseline.xlsx', index=False)
    print("\n[green]Results saved to results.csv[/green]")



if __name__ == "__main__":
    main(tyro.cli(Config))
