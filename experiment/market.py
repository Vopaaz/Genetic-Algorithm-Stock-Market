import pandas as pd
from experiment.data import read
from experiment.config import *
import glob
import os
from experiment.agent import Agent, GeneticAgent
import logging
import numpy as np
from util import KnowsFullTdf

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Stock(KnowsFullTdf):
    def __init__(self, start_date, end_date, symbol):
        self.full_tdf = read(symbol)
        self.start_date = start_date
        self.end_date = end_date


    def trade_by(self, agent):
        assert isinstance(agent, Agent)
        today = self.start_date
        agent_holding = False
        total_rev = 0
        logger.info(
            f"Trading start for {agent.__class__.__name__}{', with gene'+ str(list(agent.gene)) if isinstance(agent, GeneticAgent) else ''}"
        )
        while today < self.end_date:
            tdf = self.full_tdf.loc[: today + pd.DateOffset(days=-1)]
            decision = agent.decide(tdf)
            price = self.full_tdf.close.loc[today]
            if decision.buy() and not agent_holding:
                total_rev -= price
                agent_holding = True
            elif decision.sell() and agent_holding:
                agent_holding = False
                total_rev += price
            logger.info(f"On {today}, agent choose to {decision} on price {price}")
            today = self._next_day(today)

        price = self.full_tdf.close.loc[today]
        if agent_holding:
            total_rev += price
        logger.info(f"Trading ended. Total revenue: {total_rev}")
        return total_rev


if __name__ == "__main__":
    pass
