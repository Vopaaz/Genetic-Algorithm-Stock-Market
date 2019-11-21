import os
import random
from io import StringIO

import pandas as pd
import requests
from yaml import safe_load

API = "https://www.alphavantage.co/query"

with open("config.yml", "r", encoding="utf-8") as f:
    KEY = safe_load(f)["key"]

with open(r"pipeline/list.csv", "r", encoding="utf-8") as f:
    _raw_list = pd.read_csv(f, header=0, index_col=False)
    SYMBOL_LIST = [i for i in _raw_list.iloc[:, 0] if "$" not in i]


def get_one_stock_history(
    symbol, days=365 * 3, start_year=2016, start_month=1, start_day=1
):
    assert symbol in SYMBOL_LIST
    r = requests.get(
        API,
        params={
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "datatype": "csv",
            "apikey": KEY,
        },
    )
    start = pd.Timestamp(start_year, start_month, start_day)
    end = start + pd.DateOffset(days=days)
    df = pd.read_csv(StringIO(r.text), header=0, index_col=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df[(start <= df["timestamp"]) & (df["timestamp"] <= end)]


def save_one_stock_history(
    symbol, days=365 * 3, start_year=2016, start_month=1, start_day=1
):
    try:
        df = get_one_stock_history(symbol, days, start_year, start_month, start_day)
        with open(os.path.join("data", symbol + ".csv"), "w", encoding="utf-8") as f:
            df.to_csv(f, header=True, index=False, line_terminator="\n")
    except Exception as e:
        print(f"{symbol} failed. Error:", str(e))


def get_random_symbol(n=5):
    return random.sample(SYMBOL_LIST, n)


if __name__ == "__main__":
    for symbol in get_random_symbol():
        save_one_stock_history(symbol)
