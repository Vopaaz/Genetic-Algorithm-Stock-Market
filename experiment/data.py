import pandas as pd
import os
import glob


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
    print(type(read("CMS").index.values[0]))
    print(read("CMS").loc[:pd.Timestamp(year=2016, month=1, day=5)])
    print(read("CMS").iloc[:2])
