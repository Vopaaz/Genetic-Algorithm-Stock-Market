import sys

sys.path.append(".")

import numpy as np
import pandas as pd

from experiment.util.config import logger
import matplotlib.pyplot as plt

class Experiment(object):
    def __init__(self, population, evolution, market):
        self.population = population
        self.evolution = evolution
        self.market = market

    def train(self, epoch):
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

    def test(self, market):
        logger.info("Start testing.")
        test_eval = market.evaluate(self.population)
        self.test_max = max(test_eval)
        self.test_mean = np.mean(test_eval)
        logger.info(f"Test best: {self.test_max}, average: {self.test_mean}")

    def visualize(self, filename=None):
        assert hasattr(self, "history")
        self.history.plot()
        if hasattr(self, "test_max") and hasattr(self, "test_mean"):
            plt.axhline(self.test_max, color="green")
            plt.axhline(self.test_mean, color="burlywood")
        plt.xlabel("Epoch")
        plt.ylabel("Performance")
        if filename:
            plt.savefig(filename)
        else:
            plt.show()


if __name__ == "__main__":
    from experiment.background.agent import (
        GeneticBitAgent,
        GeneticRealAgent,
        GeneticComplexAgent,
    )
    from experiment.background.market import Market
    from experiment.GA import BitEvolution, RealEvolution, ComplexEvolution
    from experiment.util.config import TRAIN_START, TRAIN_END, TEST_START, TEST_END

    population = [GeneticComplexAgent() for _ in range(10)]
    evolution = ComplexEvolution(0.6, 0.75, 0.1)
    market = Market(TRAIN_START, TRAIN_END)
    test_market = Market(TEST_START, TEST_END)
    e = Experiment(population, evolution, market)
    e.train(3)
    e.test(test_market)
    e.visualize()
