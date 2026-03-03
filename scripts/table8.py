import tyro
import pandas as pd


from dataclasses import dataclass, field
from rich import print

from propensity_score_matching.utils import analyze_continuous_column, analyze_categorical_column, p_val_categorical, p_val_continuous, create_baseline_table, get_all_categories

@dataclass
class Config():
    file: str
    continuous_cols: list = field(default_factory=lambda: ["TE capacity", "TE initial fill", "implant size"])
    categorical_cols: list = field(default_factory=lambda: ["reoperation (no=0, yes=1)","replacement of implant/TE at same time"])
    combined_cat_cols: list = field(default_factory=lambda: ["complications_2", "complications_3", "complications_4"])

def main(cfg: Config):
    df_immediate = pd.read_excel(cfg.file, sheet_name="Treatment (Matched)")
    df_delayed = pd.read_excel(cfg.file, sheet_name="Control (Matched)")
    df_total = pd.concat([df_immediate, df_delayed], ignore_index=True)
    
    # Build results list
    results = []
    
    # Process continuous columns
    for column_name in cfg.continuous_cols:
        try:
            sheet1_result = analyze_continuous_column(df_immediate, column_name)
        except:
            sheet1_result = ''
        
        sheet2_result = analyze_continuous_column(df_delayed, column_name)
        total_result = analyze_continuous_column(df_total, column_name)

        try:
            p_value_result = p_val_continuous(df_immediate, df_delayed, column_name)
        except:
            p_value_result = ''

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
    
    #Process multiple columns as one
    # Get all unique categories across all sheets
    all_categories = get_all_categories(df_immediate, df_delayed, cfg.combined_cat_cols)
    all_categories = {str(cat) for cat in all_categories}

    sheet1_result = analyze_categorical_column(df_immediate, cfg.combined_cat_cols, all_categories)
    sheet2_result = analyze_categorical_column(df_delayed, cfg.combined_cat_cols, all_categories)
    total_result = analyze_categorical_column(df_total, cfg.combined_cat_cols, all_categories)
    p_value_result = p_val_categorical(df_immediate, df_delayed, cfg.combined_cat_cols)
    

    for i, category in enumerate(sorted(all_categories)):
        if i == 0:
            # First row: just column name and p-value, no data
            results.append({
                'col': "Complications",
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
    df_results.to_excel('data/result_table8.xlsx', index=False)
    print("\n[green]Results saved to results.csv[/green]")



if __name__ == "__main__":
    main(tyro.cli(Config))
