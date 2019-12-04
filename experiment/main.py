import sys

sys.path.append(".")

from experiment.background.agent import GeneticBitAgent, GeneticRealAgent
from experiment.background.market import Market
from experiment.GA import BitEvolution, RealEvolution
from experiment.util.config import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from experiment import Experiment


if __name__ == "__main__":
    population = [GeneticRealAgent() for _ in range(15)]
    evolution = RealEvolution(0.6, 0.75, 0.1)
    market = Market(TEST_START, TEST_END)
    test_market = Market(TEST_START, TEST_END)
    e = Experiment(population, evolution, market)
    e.train(10)
    e.test(test_market)
    e.visualize()
