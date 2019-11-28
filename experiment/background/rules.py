import sys

sys.path.append(".")

from typing import *

import pandas as pd

from experiment.background.decision import Buy, Hold, Sell


class Rule(object):
    def decide(self, tdf: pd.DataFrame) -> Union[Buy, Sell, Hold]:
        raise NotImplementedError


class SingleMACrossover(Rule):
    def __init__(self, n: int = 28):
        assert n > 1
        self.n = n

    def decide(self, tdf):
        ts = tdf.close
        ma_today = ts.iloc[-1 - self.n : -1].mean()
        ma_yesterday = ts.iloc[-2 - self.n : -2].mean()

        today = ts.iloc[-1]
        yesterday = ts.iloc[-2]

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
        ma_short_today = ts.iloc[-1 - self.short_n : -1].mean()
        ma_short_yesterday = ts.iloc[-2 - self.short_n : -2].mean()

        ma_long_today = ts.iloc[-1 - self.long_n : -1].mean()
        ma_long_yesterday = ts.iloc[-2 - self.long_n : -2].mean()

        if ma_short_today > ma_long_today and ma_short_yesterday < ma_long_yesterday:
            return Buy()
        elif ma_short_today < ma_long_today and ma_short_yesterday > ma_long_yesterday:
            return Sell()
        else:
            return Hold()


class RelativeStrengthIndex(Rule):
    def __init__(
        self,
        n: int = 14,
        buy_signal: int = 30,
        sell_signal: int = 70,
        avg_method: int = 0,
    ):
        """Implementation of Relative Strength Index

        Args:
            n (int): Number of days
            buy_signal (int): When RSI reaches this, buy
            sell_signal (int): When RSI reaches this, sell
            avg_method (int):
                - 0: Simple Moving Average
                - 1: Exponential Moving Average
                - 2: Wilder's Smoothing Method
                See https://www.macroption.com/rsi-calculation/ for detail.
        """
        assert n > 1
        assert buy_signal < sell_signal
        assert avg_method in [0, 1]
        self.n = n
        self.buy_signal = buy_signal
        self.sell_signal = sell_signal
        self.avg_method = avg_method

    def _compute_RSI(self, tdf):
        ts = tdf.close.iloc[-1 - self.n : -1].reset_index(drop=True)
        ts_lag = tdf.close.iloc[-2 - self.n : -2].reset_index(drop=True)
        diff = ts - ts_lag
        up = diff[diff > 0]
        down = -diff[diff < 0]

        if self.avg_method == 0:
            avg_up = up.sum() / self.n
            avg_down = down.sum() / self.n
        else:
            raise NotImplementedError

        RS = avg_up / avg_down
        return 100 - 100 / (1 + RS)

    def decide(self, tdf):
        RSI = self._compute_RSI(tdf)
        if RSI < self.buy_signal:
            return Buy()
        elif RSI > self.sell_signal:
            return Sell()
        else:
            return Hold()


class StochasticOscillator(Rule):
    def __init__(self, n: int = 14, buy_signal: int = 20, sell_signal: int = 80):
        assert n > 1
        assert buy_signal < sell_signal
        self.n = n
        self.buy_signal = buy_signal
        self.sell_signal = sell_signal

    def decide(self, tdf):
        ts = tdf.close
        C = ts.iloc[-1]
        H = ts.iloc[-1 - self.n : -1].max()
        L = ts.iloc[-1 - self.n : -1].min()
        K = (C - L) / (H - L) * 100
        if K < self.buy_signal:
            return Buy()
        elif K > self.sell_signal:
            return Sell()
        else:
            return Hold()


class MA918(DoubleMACrossover):
    def __init__(self):
        super().__init__(9, 18)


class MA4918(Rule):
    def __init__(self):
        self.m1 = DoubleMACrossover(4,9)
        self.m2 = DoubleMACrossover(9,18)

    def decide(self, tdf):
        d1 = self.m1.decide(tdf)
        d2 = self.m2.decide(tdf)
        if d1.sell() and d2.sell():
            return Sell()
        elif d1.buy() and d2.buy():
            return Buy()
        else:
            return Hold()




if __name__ == "__main__":
    from experiment.util.data import read

    df = read("CMS")
    # print(SingleMACrossover().decide(df))

    print(RelativeStrengthIndex()._compute_RSI(df))
    MA918()
