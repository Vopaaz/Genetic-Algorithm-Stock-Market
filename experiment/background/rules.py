import sys

sys.path.append(".")

from typing import *

import pandas as pd
import numpy as np

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
        ma_today = ts.iloc[-self.n :].mean()
        ma_yesterday = ts.iloc[-1 - self.n : -1].mean()

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
        ma_short_today = ts.iloc[-self.short_n :].mean()
        ma_short_yesterday = ts.iloc[-1 - self.short_n : -1].mean()

        ma_long_today = ts.iloc[-self.long_n :].mean()
        ma_long_yesterday = ts.iloc[-1 - self.long_n : -1].mean()

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

    def decide(self, tdf):
        RSI = _compute_RSI(tdf.close, self.n, self.avg_method)
        if RSI < self.buy_signal:
            return Buy()
        elif RSI > self.sell_signal:
            return Sell()
        else:
            return Hold()


def _compute_RSI(ts, n, avg_method=0):
    _ts = ts.copy(deep=True)
    ts = _ts.iloc[-n:].reset_index(drop=True)
    ts_lag = _ts.iloc[-1 - n : -1].reset_index(drop=True)
    diff = ts - ts_lag
    up = diff[diff > 0]
    down = -diff[diff < 0]

    if avg_method == 0:
        avg_up = up.sum() / n
        avg_down = down.sum() / n
    else:
        raise NotImplementedError

    if avg_down == 0:
        return 100
    else:
        RS = avg_up / avg_down
        return 100 - 100 / (1 + RS)


class StochasticOscillator(Rule):
    def __init__(self, n: int = 14, buy_signal: int = 20, sell_signal: int = 80):
        assert n > 1
        assert buy_signal < sell_signal
        self.n = n
        self.buy_signal = buy_signal
        self.sell_signal = sell_signal

    def decide(self, tdf):
        C = tdf.close.iloc[-1]
        H = tdf.high.iloc[-self.n :].max()
        L = tdf.low.iloc[-self.n :].min()
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
        self.m1 = DoubleMACrossover(4, 9)
        self.m2 = DoubleMACrossover(9, 18)

    def decide(self, tdf):
        d1 = self.m1.decide(tdf)
        d2 = self.m2.decide(tdf)
        if d1.sell() and d2.sell():
            return Sell()
        elif d1.buy() and d2.buy():
            return Buy()
        else:
            return Hold()


class MACD(Rule):
    def __init__(self, short_n: int = 12, long_n: int = 26, signal: float = 0):
        assert 1 < short_n < long_n
        self.short_n = short_n
        self.long_n = long_n
        self.signal = signal

    def decide(self, tdf):
        ts = tdf.close
        short_EMA = self._compute_EMA(ts, self.short_n)
        long_EMA = self._compute_EMA(ts, self.long_n)
        MACD_today = short_EMA - long_EMA

        ts_yesterday = ts.iloc[:-1]
        short_EMA = self._compute_EMA(ts_yesterday, self.short_n)
        long_EMA = self._compute_EMA(ts_yesterday, self.long_n)
        MACD_yesterday = short_EMA - long_EMA

        if MACD_yesterday < self.signal and MACD_today > self.signal:
            return Buy()
        elif MACD_yesterday > self.signal and MACD_today < self.signal:
            return Sell()
        else:
            return Hold()

    def _compute_EMA(self, ts, n):
        return ts.iloc[-n:].ewm(span=n).mean().iloc[-1]


class MoneyFlowIndex(Rule):
    def __init__(self, n: int = 14, buy_signal: int = 10, sell_signal: int = 90):
        assert n > 1
        assert buy_signal < sell_signal
        self.n = n
        self.buy_signal = buy_signal
        self.sell_signal = sell_signal

    def decide(self, tdf):
        today_MFI = self._compute_MFI(tdf)
        yesterday_MFI = self._compute_MFI(tdf.iloc[:-1])

        if today_MFI > self.sell_signal:
            return Sell()
        elif today_MFI < self.buy_signal:
            return Buy()
        else:
            return Hold()

    def _compute_MFI(self, _tdf):

        tdf = _tdf.iloc[-self.n :].copy(deep=True)
        tdf_lag = _tdf.iloc[-1 - self.n : -1].copy(deep=True)

        typical_price = ((tdf.high + tdf.low + tdf.close) / 3).values
        typical_price_lag = ((tdf_lag.high + tdf_lag.low + tdf_lag.close) / 3).values

        typical_price_symbol = np.vectorize(lambda x: 1 if x > 0 else -1)(
            typical_price - typical_price_lag
        )
        typical_price_symbol[0] = 0

        raw_money_flow = typical_price_symbol * typical_price * tdf.volume.values

        positive_flow = raw_money_flow[raw_money_flow > 0].sum()
        negative_flow = -raw_money_flow[raw_money_flow < 0].sum()
        money_ratio = positive_flow / negative_flow

        MFI = 100 - 100 / (1 + money_ratio)
        return MFI


class CommodityChannelIndex(Rule):
    def __init__(
        self,
        CCI_n: int = 20,
        trend_n: int = 5,
        sell_signal: int = -100,
        buy_signal: int = 100,
    ):
        assert CCI_n > 1
        assert trend_n > 1
        assert sell_signal < buy_signal
        self.CCI_n = CCI_n
        self.trend_n = trend_n
        self.sell_signal = sell_signal
        self.buy_signal = buy_signal

    def decide(self, tdf):
        now_CCI = self._compute_CCI(tdf)
        old_CCIs = [self._compute_CCI(tdf.iloc[:-i]) for i in range(1, self.trend_n)]
        if now_CCI > self.buy_signal and now_CCI > max(old_CCIs):
            return Buy()
        elif now_CCI < self.sell_signal and now_CCI > min(old_CCIs):
            return Sell()
        else:
            return Hold()

    def _compute_CCI(self, _tdf):
        tdf = _tdf.iloc[-self.CCI_n :].copy(deep=True)
        typical_price = (tdf.high + tdf.low + tdf.close) / 3
        p = typical_price.iloc[-1]
        SMA = typical_price.mean()
        MD = typical_price.mad()
        return (p - SMA) / (MD * 0.015)


class StochasticRSI(Rule):
    def __init__(self, n: int = 14, buy_signal: float = 0.2, sell_signal: float = 0.8):
        assert n > 1
        assert buy_signal < sell_signal
        self.n = n
        self.buy_signal = buy_signal
        self.sell_signal = sell_signal

    def decide(self, tdf):
        stochRSI = self._compute_stochRSI(tdf.close)
        if stochRSI < self.buy_signal:
            return Buy()
        elif stochRSI > self.sell_signal:
            return Sell()
        else:
            return Hold()

    def _compute_stochRSI(self, ts):
        now_RSI = _compute_RSI(ts, self.n)
        old_RSIs = [_compute_RSI(ts.iloc[:-i], self.n) for i in range(1, self.n)]
        min_RSI = min(old_RSIs)
        max_RSI = max(old_RSIs)
        return (now_RSI - min_RSI) / (max_RSI - min_RSI)


if __name__ == "__main__":
    from experiment.util.data import read

    df = read("CMS")
    # print(SingleMACrossover().decide(df))
    # print(RelativeStrengthIndex().decide(df))
    print(_compute_RSI(df.close, 15))
    # MA918()
    # print(CommodityChannelIndex()._compute_CCI(df.iloc[:-15]))
    print(StochasticRSI()._compute_stochRSI(df.close.iloc[:-11]))
    print(StochasticRSI().decide(df.iloc[:-11]))
