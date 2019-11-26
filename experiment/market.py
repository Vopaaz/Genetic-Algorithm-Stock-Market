import pandas as pd
from experiment.data import read
from experiment.config import *
import glob
import os
from experiment.agent import Agent


class Stock(object):
    def __init__(self, start_date, end_date, symbol):
        self.full_tdf = read(symbol)
        self.start_date = start_date
        self.end_date = end_date

    def __next_day(self, day):
        day += pd.DateOffset(day=1)
        while day not in self.full_tdf.index.values:
            day += pd.DateOffset(day=1)

    def trade(self, agent):
        assert isinstance(agent, Agent)
        today = self.start_date
        agent_holding = False
        total_rev = 0

        while today < self.end_date:
            decision = agent.decide(self.full_tdf.loc[: today + pd.DateOffset(day=-1)])
            price = self.full_tdf.close.loc[today]
            if decision.buy() and not agent_holding:
                total_rev -= price
            elif decision.sell() and agent_holding:
                total_rev += price

        price = self.full_tdf.close.loc[today]
        if agent_holding:
            total_rev += price

        return total_rev


if __name__ == "__main__":
    pass
