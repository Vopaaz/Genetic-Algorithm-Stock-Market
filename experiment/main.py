import sys
sys.path.append(".")

from experiment.market import Stock
from experiment.agent import GeneticBitAgent, BenchmarkAgent
from experiment.config import TRAIN_START, TRAIN_END
from experiment.data import read




if __name__ == "__main__":
    SYMBOL = "CMS"
    a = GeneticBitAgent()
    b = BenchmarkAgent(SYMBOL)
    m = Stock(TRAIN_START, TRAIN_END, SYMBOL)
    print(m.trade_by(a))
    print(m.trade_by(b))

