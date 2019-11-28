import sys

sys.path.append(".")

import numpy as np
import pandas as pd

from experiment.util.config import logger


class Experiment(object):
    def __init__(self, population, evolution, market):
        self.population = population
        self.evolution = evolution
        self.market = market

    def run(self, epoch):
        logger.info("Experiment start.")
        population = self.population
        result = []
        for i in range(epoch):
            logger.info(f"Epoch {i+1}/{epoch} start.")
            evaluation = self.market.evaluate(population)
            best = max(evaluation)
            avg = np.mean(evaluation)
            result.append([best, avg])
            logger.info(f"Epoch {i+1}'s best: {best}, average: {avg}")
            population = self.evolution.evolve(population, evaluation)

        self.population = population
        self.history = pd.DataFrame(result, columns=["best", "avg"])
        return self.history

if __name__ == "__main__":
    from experiment.background.agent import GeneticBitAgent
    from experiment.background.market import Market
    from experiment.GA import BitEvolution
    from experiment.util.config import TRAIN_START, TRAIN_END
    import matplotlib.pyplot as plt

    population = [GeneticBitAgent() for _ in range(5)]
    evolution = BitEvolution(0.8, 0.8, 0.2)
    market = Market(TRAIN_START, TRAIN_END)
    e = Experiment(population, evolution, market)
    res= e.run(5)
    res.plot()
    plt.show()

    '''
    Problem in implementation: too slow.
    For 10 genetic bit agent, evaluation takes about 3 min.
    If a 100 agent and 50 epoch is used, it will take 1500 min,
    which is 25 hours to complete. This is unacceptable.
    With adding more financial rules, this time will grow even larger.
    Optimization needs to be done.
    '''
