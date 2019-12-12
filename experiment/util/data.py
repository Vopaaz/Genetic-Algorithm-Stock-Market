import glob
import os

import pandas as pd


def read(symbol):
    path = os.path.join(".", "data", f"{symbol}.csv")
    with open(path, "r", encoding="utf-8") as f:
        df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df.iloc[::-1]


ALL_SYMBOLS = [
    os.path.splitext(os.path.basename(f))[0] for f in glob.glob(r"data/*.csv")
]

if __name__ == "__main__":
    print(read("CMS").loc[: pd.Timestamp(year=2016, month=1, day=5)])
    print(read("CMS").iloc[:2])
    print(type(read("CMS").index[0]))
