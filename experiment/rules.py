import sys

sys.path.append(".")

import pandas as pd
from typing import *
from experiment.decision import Buy, Sell, Hold


class Rule(object):
    def decide(self, tdf: pd.DataFrame) -> Union[Buy, Sell, Hold]:
        raise NotImplementedError


class SingleMACrossover(Rule):
    def __init__(self, n: int = 28):
        assert n > 1
        self.n = n

    def decide(self, tdf):
        ts = tdf.close
        ma_today = ts.iloc[0 : self.n].mean()
        ma_yesterday = ts.iloc[1 : self.n + 1].mean()

        today = ts.iloc[0]
        yesterday = ts.iloc[1]

        if today > ma_today and yesterday < ma_yesterday:
            return Buy()
        elif today < ma_today and yesterday > ma_yesterday:
            return Sell()
        else:
            return Hold()


class DoubleMACrossover(Rule):
    def __init__(self, short_n: int = 25, long_n: int = 50):
        assert 1 < short_n < long_n
        self.short_n = short_n
        self.long_n = long_n

    def decide(self, tdf):
        ts = tdf.close
        ma_short_today = ts.iloc[0 : self.short_n].mean()
        ma_short_yesterday = ts.iloc[1 : self.short_n + 1].mean()

        ma_long_today = ts.iloc[0 : self.long_n].mean()
        ma_long_yesterday = ts.iloc[1 : self.long_n + 1].mean()

        if ma_short_today > ma_long_today and ma_short_yesterday < ma_long_yesterday:
            return Buy()
        elif ma_short_today < ma_long_today and ma_short_yesterday > ma_long_yesterday:
            return Sell()
        else:
            return Hold()


if __name__ == "__main__":
    from experiment.data import read

    df = read("CMS")
    print(SingleMACrossover().decide(df))
