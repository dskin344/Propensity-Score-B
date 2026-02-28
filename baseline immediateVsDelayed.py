

import tyro
import pandas as pd


from dataclasses import dataclass, field
from rich import print



@dataclass
class Config():
    file: str


def main(cfg: Config):
    df_immediate = pd.read_excel(cfg.file, sheet_name="single stage")
    df_delayed = pd.read_excel(cfg.file, sheet_name="two-stage")

    print(df_immediate.columns)
    print(df_delayed.columns)


if __name__ == "__main__":
    main(tyro.cli(Config))
