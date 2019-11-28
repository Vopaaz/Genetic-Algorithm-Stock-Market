import sys

sys.path.append(".")

from experiment.background.agent import BenchmarkAgent, GeneticBitAgent
from experiment.util.config import TRAIN_END, TRAIN_START
from experiment.util.data import read
from experiment.background.market import Market



if __name__ == "__main__":
    a = GeneticBitAgent([0, 0, 1])
    m = Market(TRAIN_START, TRAIN_END)
    print(m.evaluate([a]))
