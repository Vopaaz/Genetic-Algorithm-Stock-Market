import sys

sys.path.append(".")

import pandas as pd
from typing import *
from experiment.decision import Buy, Sell, Hold, make_decision
from experiment.rules import *
import numpy as np


class Agent(object):

    RULES = [SingleMACrossover, DoubleMACrossover]

    def __init__(self):
        self.init_rules()

    def init_rules(self):
        raise NotImplementedError

    def decide(self, tdf: pd.DataFrame) -> Union[Buy, Sell, Hold]:
        raise NotImplementedError


class GeneticBitAgent(Agent):
    def init_rules(self):
        self.rules = [rule() for rule in self.RULES]

    def __init__(self, gene=None):
        if gene is None:
            self.gene = np.random.randint(0, 1, len(self.RULES))
        elif isinstance(gene, list):
            assert len(gene) == len(self.RULES)
            self.gene = np.array(gene)
        elif isinstance(gene, np.ndarray):
            assert len(gene) == len(self.RULES)
            self.gene = gene
        else:
            raise TypeError(f"{type(gene)} is not a valid type for gene.")
        super().__init__()

    def decide(self, tdf):
        rule_decisions = np.array([rule.decide(tdf) for rule in self.rules])
        vote_decision = (rule_decisions * self.gene).sum()
        return make_decision(vote_decision)


class BenchMarkAgent(Agent):
    def __init__(self, full_tdf):
        self.full_tdf = full_tdf

    def decide(self, tdf):
        today = tdf.index.values[0]

        tomorrow = today + pd.DateOffset(day=1)
        while tomorrow not in self.full_tdf.index.values:
            tomorrow += pd.DateOffset(day=1)

        ts = self.full_tdf.close
        if ts.loc[today] > ts.loc[tomorrow]:
            return Sell()
        elif ts.loc[today] < ts.loc[tomorrow]:
            return Buy()
        else:
            return Hold()


if __name__ == "__main__":
    from experiment.data import read

    tdf = read("CMS")
    a = GeneticBitAgent()
    print(a.decide(tdf))
