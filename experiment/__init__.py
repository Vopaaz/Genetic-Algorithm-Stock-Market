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
            evaluation = self.market.evaluate(population)
            best = max(evaluation)
            avg = np.mean(evaluation)
            result.append([best, avg])
            logger.info(f"Generation {i}'s best: {best}, average: {avg}")

            logger.info(f"Start evolving generation {i+1}/{epoch}.")
            population = self.evolution.evolve(population, evaluation)

        self.population = population
        logger.info(f"Final population generated, start evaluation.")
        self.train_eval = self.market.evaluate(population)
        best = max(self.train_eval)
        avg = np.mean(self.train_eval)
        result.append([best, avg])
        logger.info(f"Final population's best: {best}, average: {avg}")

        self.history = pd.DataFrame(result, columns=["best", "avg"])
        return self.history

    def visualize(self):
        assert hasattr(self, "history")
        self.history.plot()
        plt.xlabel("Epoch")
        plt.ylabel("Performance")
        plt.show()


if __name__ == "__main__":
    from experiment.background.agent import GeneticBitAgent
    from experiment.background.market import Market
    from experiment.GA import BitEvolution
    from experiment.util.config import TRAIN_START, TRAIN_END
    import matplotlib.pyplot as plt

    population = [GeneticBitAgent() for _ in range(10)]
    evolution = BitEvolution(0.5, 0.6, 0.1)
    market = Market(TRAIN_START, TRAIN_END)
    e = Experiment(population, evolution, market)
    e.run(2)
    e.visualize()

    """
    Problem in implementation: too slow.
    For 10 genetic bit agent, evaluation takes about 3 min.
    If a 100 agent and 50 epoch is used, it will take 1500 min,
    which is 25 hours to complete. This is unacceptable.
    With adding more financial rules, this time will grow even larger.
    Optimization needs to be done.
    """
