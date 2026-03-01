import tyro
import pandas as pd


from dataclasses import dataclass
from rich.table import Table
from rich import print

from utils import analyze_continuous_column, analyze_categorical_column, calculate_p_value

@dataclass
class Config():
    file: str
    continuous_cols: list = ["age", "BMI"]
    categorical_cols: list = ["raceethnic", "diabetes", "HTN", "SPY",
                     "tobacoo_history", "alcohol_history", "pre-pec", "sub-pec",
                     "NSM (mastectomy type)", "SSM (mastectomy type)", "PRE  chemotherapy (yes=1)",
                     "POST  chemotherapy (yes=1)", "immunotherapy (keytruda?)", 
                     "RT (yes=1)", "adjuvant endocrine", "ADM/dermal sling (type?)",
                     "SLNB (yes=1)", "ALND (yes=1)", "ER+", "PR+", "HER2+", "grade1", 
                     "mastectomy laterality", "cancer laterality R(0), L (1), both (2)",
                     "clinical stage"]

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
        p_value_result = calculate_p_value(df_immediate[column_name], df_delayed[column_name])
        
        results.append({
            'Column': column_name,
            'Type': 'Continuous',
            'Sheet 1': sheet1_result['central_tendency_value'],
            'Sheet 2': sheet2_result['central_tendency_value'],
            'Total': total_result['central_tendency_value'],
            'P-Value': p_value_result['p_value'],
            'Category': None
        })
    
    # Process categorical columns
    for column_name in cfg.categorical_cols:
        sheet1_result = analyze_categorical_column(df_immediate, column_name)
        sheet2_result = analyze_categorical_column(df_delayed, column_name)
        total_result = analyze_categorical_column(df_total, column_name)
        
        # Get all unique categories across all sheets
        all_categories = set(df_sheet1[column_name].unique()) | set(df_delayed[column_name].unique())
        
        for i, category in enumerate(sorted(all_categories)):
            # Count occurrences for each category in each sheet
            sheet1_count = (df_sheet1[column_name] == category).sum()
            sheet2_count = (df_sheet2[column_name] == category).sum()
            total_count = (df_total[column_name] == category).sum()
            
            # Calculate percentages
            sheet1_pct = (sheet1_count / len(df_sheet1)) * 100
            sheet2_pct = (sheet2_count / len(df_sheet2)) * 100
            total_pct = (total_count / len(df_total)) * 100
            
            results.append({
                'Column': column_name if i == 0 else '',
                'Type': 'Categorical',
                'Sheet 1': f"{sheet1_count} ({sheet1_pct:.1f}%)",
                'Sheet 2': f"{sheet2_count} ({sheet2_pct:.1f}%)",
                'Total': f"{total_count} ({total_pct:.1f}%)",
                'P-Value': 'N/A',
                'Category': category
            })
    
    # Build Rich table
    table = Table(title="Analysis Results")
    table.add_column("Column", style="cyan")
    table.add_column("Category", style="blue")
    table.add_column("Sheet 1", style="magenta")
    table.add_column("Sheet 2", style="magenta")
    table.add_column("Total", style="green")
    table.add_column("P-Value", style="yellow")
    
    for _, row in results:
        table.add_row(
            str(row['Column']),
            str(row['Category'] if row['Category'] else ''),
            str(row['Sheet 1']),
            str(row['Sheet 2']),
            str(row['Total']),
            str(row['P-Value'])
        )
    
    print(table)
    
    # # Save results
    # df_results = pd.DataFrame(results)
    # df_results.to_excel('data/results.xlsx', index=False)
    # print("\n[green]Results saved to results.csv[/green]")



if __name__ == "__main__":
    main(tyro.cli(Config))
