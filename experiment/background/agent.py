import sys

sys.path.append(".")

from typing import *

import numpy as np
import pandas as pd

from experiment.background.decision import Buy, Hold, Sell, make_decision
from experiment.background.rules import *
from experiment.background.util import KnowsFullTdf
from experiment.util.data import read


class Agent(object):
    def decide(self, tdf: pd.DataFrame) -> Union[Buy, Sell, Hold]:
        raise NotImplementedError


class GeneticAgent(Agent):

    RULES = [
        SingleMACrossover,
        DoubleMACrossover,
        RelativeStrengthIndex,
        StochasticOscillator,
        MA918,
        MA4918,
        MACD,
        MoneyFlowIndex,
        CommodityChannelIndex,
        StochasticRSI,
    ]

    def __init__(self):
        self.init_rules()

    def _init_with_gene(self, gene):
        if isinstance(gene, list):
            assert len(gene) == len(self.RULES)
            self.gene = np.array(gene)
        elif isinstance(gene, np.ndarray):
            assert len(gene) == len(self.RULES)
            self.gene = gene
        else:
            raise TypeError(f"{type(gene)} is not a valid type for gene.")

    def init_rules(self):
        raise NotImplementedError

    def decide(self, tdf):
        rule_decisions = np.array([rule.decide(tdf) for rule in self.rules])
        vote_decision = (rule_decisions * self.gene).sum()
        return make_decision(vote_decision)


class GeneticSimpleAgent(GeneticAgent):
    def init_rules(self):
        self.rules = [rule() for rule in self.RULES]


class GeneticBitAgent(GeneticSimpleAgent):
    def __init__(self, gene=None):
        if gene is None:
            self.gene = np.random.randint(0, 2, len(self.RULES))
        else:
            self._init_with_gene(gene)
        super().__init__()


class GeneticRealAgent(GeneticSimpleAgent):
    def __init__(self, gene=None):
        if gene is None:
            self.gene = np.random.random(len(self.RULES))
        else:
            self._init_with_gene(gene)
        super().__init__()


class GeneticComplexAgent(GeneticAgent):
    def __init__(self, gene=None, param_gene=None):
        if gene is None:
            self.gene = np.random.random(len(self.RULES))
        else:
            self._init_with_gene(gene)

        if param_gene is None:
            self.param_gene = [rule.generate_param() for rule in self.RULES]
        else:
            self.param_gene = param_gene
        super().__init__()

    def init_rules(self):
        self.rules = [rule(*gene) for rule, gene in zip(self.RULES, self.param_gene)]


class BenchmarkAgent(Agent, KnowsFullTdf):
    def __init__(self, symbol):
        self.full_tdf = read(symbol)

    def decide(self, tdf):
        today = tdf.index[-1]
        tomorrow = self._next_day(today)

        ts = self.full_tdf.close
        if ts.loc[today] > ts.loc[tomorrow]:
            return Sell()
        elif ts.loc[today] < ts.loc[tomorrow]:
            return Buy()
        else:
            return Hold()


if __name__ == "__main__":
    from experiment.util.data import read

    tdf = read("CMS")
    a = GeneticBitAgent()
    print(a.decide(tdf))

    # b = GeneticComplexAgent()
    # print(b.decide(tdf))
