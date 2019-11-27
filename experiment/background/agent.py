import sys

sys.path.append(".")

import pandas as pd
from typing import *
from experiment.background.decision import Buy, Sell, Hold, make_decision
from experiment.background.rules import *
from experiment.util.data import read
import numpy as np
from experiment.background.util import KnowsFullTdf


class Agent(object):
    def decide(self, tdf: pd.DataFrame) -> Union[Buy, Sell, Hold]:
        raise NotImplementedError


class GeneticAgent(Agent):

    RULES = [SingleMACrossover, DoubleMACrossover]

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


class GeneticSimpleAgent(GeneticAgent):
    def init_rules(self):
        self.rules = [rule() for rule in self.RULES]

    def decide(self, tdf):
        rule_decisions = np.array([rule.decide(tdf) for rule in self.rules])
        vote_decision = (rule_decisions * self.gene).sum()
        return make_decision(vote_decision)


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
            arr = np.random.random(len(self.RULES))
            self.gene = arr / arr.sum()
        else:
            self._init_with_gene(gene)
        super().__init__()


# Problem in GeneticComplexAgent:
# Hyperparameters have constraint, for example, in doubleMA, short_n < long_n.
# Therefore we can't directly use random number for parameter genes
# Ways to address:
#   1. when initialize, loop until valid
#       Problem: when mutate, invalid parameter genes could still be generated
#   2. change the implementation of rules, for example make the doubleMA automatically switch
#           short_n and long_n when it found short_n > long_n
#       Problem: do not know if this is applicable in every rule

# class GeneticComplexAgent(GeneticAgent):
#     def __init__(self, gene=None):
#         if gene is None:
#             arr = np.random.random(len(self.RULES))
#             self.gene = arr / arr.sum()
#             self.param_gene = [
#                 list(np.random.randint(5, 100, rule.N_PARAM)) for rule in self.RULES
#             ]
#         else:
#             self._init_with_gene(gene)
#         super().__init__()

#     def init_rules(self):
#         self.rules = [rule(*gene) for rule, gene in zip(self.RULES, self.param_gene)]


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
