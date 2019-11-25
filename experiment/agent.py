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
        self.rules = [rule() for rule in self.RULES]

    def decide(self, tdf: pd.DataFrame) -> Union[Buy, Sell, Hold]:
        raise NotImplementedError


class GeneticBitAgent(Agent):
    def __init__(self, gene=None):
        if gene is None:
            self.gene = np.random.random(len(self.RULES))
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
    pass


if __name__ == "__main__":
    from experiment.data import read

    tdf = read("CMS")
    a = GeneticBitAgent()
    print(a.decide(tdf))
