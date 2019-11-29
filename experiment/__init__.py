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
    from experiment.background.agent import GeneticBitAgent, GeneticRealAgent
    from experiment.background.market import Market
    from experiment.GA import BitEvolution, RealEvolution
    from experiment.util.config import TRAIN_START, TRAIN_END
    import matplotlib.pyplot as plt

    population = [GeneticRealAgent() for _ in range(15)]
    evolution = RealEvolution(0.6, 0.75, 0.5)
    market = Market(TRAIN_START, TRAIN_END)
    e = Experiment(population, evolution, market)
    e.run(5)
    e.visualize()
