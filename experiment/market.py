import pandas as pd
from experiment.data import read, ALL_SYMBOLS
from experiment.config import *
import glob
import os
from experiment.agent import Agent, GeneticAgent, BenchmarkAgent
import logging
import numpy as np
from experiment.util import KnowsFullTdf

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Market(KnowsFullTdf):
    def __init__(self, start_date, end_date, all_symbols=ALL_SYMBOLS):
        self.stocks = [Stock(start_date, end_date, symbol) for symbol in all_symbols]
        self.benchmark = [
            stock.trade_by(agent)
            for stock, agent in zip(
                self.stocks, [BenchmarkAgent(symbol) for symbol in all_symbols]
            )
        ]

    def trade_by(self, agents):
        assert isinstance(agents, list)
        return [[stock.trade_by(agent) for stock in self.stocks] for agent in agents]

    def evaluate(self, agents):
        assert isinstance(agents, list)
        return [
            (np.array(raw_result) / self.benchmark).mean()
            for raw_result in self.trade_by(agents)
        ]


class Stock(KnowsFullTdf):
    def __init__(self, start_date, end_date, symbol):
        self.full_tdf = read(symbol)
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol

    def trade_by(self, agent):
        assert isinstance(agent, Agent)
        today = self.start_date
        agent_holding = False
        total_rev = 0
        logger.info(
            f"Trading '{self.symbol}' start for {agent.__class__.__name__}{', with gene'+ str(list(agent.gene)) if isinstance(agent, GeneticAgent) else ''}"
        )
        while today < self.end_date:
            tdf = self.full_tdf.loc[:today]
            decision = agent.decide(tdf)
            price = self.full_tdf.close.loc[today]
            if decision.buy() and not agent_holding:
                total_rev -= price
                agent_holding = True
            elif decision.sell() and agent_holding:
                agent_holding = False
                total_rev += price
            logger.debug(f"On {today}, agent choose to {decision} on price {price}")
            today = self._next_day(today)

        price = self.full_tdf.close.loc[today]
        if agent_holding:
            total_rev += price
        logger.info(f"Trading ended. Total revenue: {total_rev}")
        return total_rev


if __name__ == "__main__":
    SYMBOL = "CMS"
    a = BenchmarkAgent(SYMBOL)
    m = Market(TRAIN_START, TRAIN_END, [SYMBOL])
    assert m.evaluate([a])[0] == 1
