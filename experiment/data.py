import pandas as pd
import os


def read(symbol):
    path = os.path.join(".", "data", f"{symbol}.csv")
    with open(path, "r", encoding="utf-8") as f:
        df = pd.read_csv(path, index_col=0)
    return df
