import sys

sys.path.append(".")

import numpy as np
import pandas as pd

from experiment.background.agent import (
    GeneticAgent,
    GeneticBitAgent,
    GeneticRealAgent,
    GeneticSimpleAgent,
)
from experiment.util.config import logger


class Evolution(object):
    def __init__(
        self,
        survival_rate: float,
        crossover_rate: float,
        mutation_rate: float,
        elitism_rate: float = 0,
        mutation_bitwise_rate: float = 0.5,
    ):
        assert 0 < survival_rate < 1
        assert 0 < crossover_rate < 1
        assert 0 < mutation_rate < 1
        assert 0 <= elitism_rate < 1
        assert 0 < mutation_bitwise_rate < 1

        self.survival_rate = survival_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.mutation_bitwise_rate = mutation_bitwise_rate

    def evolve(self, population, evaluation):
        assert len(population) == len(evaluation)
        logger.info(f"Evolution started.")
        res = self._evolve_zipped(list(zip(population, evaluation)))
        assert len(population) == len(res)
        logger.info(f"Evolution completed.")
        return res

    def _evolve_zipped(self, population):
        original_num = len(population)
        survived = self._select(population, self.survival_rate)
        elite = self._select(population, self.elitism_rate)
        parents, non_parents = self._split_survived(survived)
        to_mutate = non_parents + self._crossover(
            parents, original_num - len(non_parents) - len(elite)
        )
        next_gen = elite + self._mutate(to_mutate)
        assert len(next_gen) == original_num
        return next_gen

    def _split_survived(self, survived):
        """Split the survived agents into crossover parents and those who goes to the next generation directly.
        Args:
            survived (list): list of agents

        Returns:
            tuple: #1 is the crossover parents and #2 shall go to the next generation.
        """
        arr = list(np.random.permutation(survived))
        point = int(len(arr) * self.crossover_rate)
        return arr[:point], arr[point:]

    def _select(self, population, rate):
        return [
            pair[0]
            for pair in sorted(population, reverse=True, key=lambda x: x[1])[
                : int(len(population) * rate)
            ]
        ]

    def _crossover(self, parents, target_num):
        assert len(parents) >= 2

        return [
            self._crossover_inner(*np.random.choice(parents, 2, replace=False))
            for _ in range(target_num)
        ]

    def _mutate(self, agents):
        agents = list(np.random.permutation(agents))
        point = int(len(agents) * self.mutation_rate)
        mutate, keep = agents[:point], agents[point:]
        return keep + [self._mutate_agent(agent) for agent in mutate]


class SimpleEvolution(Evolution):
    def _crossover_inner(self, p1, p2):
        assert type(p1) == type(p2)
        g1, g2 = p1.gene, p2.gene
        new_agent = p1.__class__(
            [(g1[i] if np.random.randint(0, 2) == 0 else g2[i]) for i in range(len(g1))]
        )
        logger.debug(
            f"Crossover of agents with gene '{g1}' and '{g2}' produces new agent with gene '{new_agent.gene}'."
        )
        return new_agent


class BitEvolution(SimpleEvolution):
    def _mutate_agent(self, agent):
        new_agent = agent.__class__(
            [
                (
                    bit
                    if np.random.uniform() < self.mutation_bitwise_rate
                    else np.random.randint(0, 2)
                )
                for bit in agent.gene
            ]
        )
        logger.debug(
            f"Mutating agent's gene from '{agent.gene}' to '{new_agent.gene}' "
        )
        return new_agent


class RealEvolution(SimpleEvolution):
    def _mutate_agent(self, agent):
        new_agent = agent.__class__(
            [
                (
                    real
                    if np.random.uniform() < self.mutation_bitwise_rate
                    else np.random.uniform()
                )
                for real in agent.gene
            ]
        )
        logger.debug(
            f"Mutating agent's gene from '{agent.gene}' to '{new_agent.gene}' "
        )
        return new_agent


class ComplexEvolution(Evolution):
    def _crossover_inner(self, p1, p2):
        assert type(p1) == type(p2)
        gene1, gene2 = p1.gene, p2.gene
        param1, param2 = p1.param_gene, p2.param_gene
        assert len(gene1) == len(gene2) == len(param1) == len(param2)

        choice = np.random.randint(0, 2, len(gene1))
        new_agent = p1.__class__(
            [(gene1[i] if choice[i] == 0 else gene2[i]) for i in range(len(choice))],
            [(param1[i] if choice[i] == 0 else param2[i]) for i in range(len(choice))],
        )
        logger.debug(
            f"Crossover of agents with gene '{gene1}', '{param1}' and '{gene2}', '{param2}' produces new agent with gene '{new_agent.gene}', '{new_agent.param_gene}'."
        )
        return new_agent

    def _mutate_agent(self, agent):
        choice = np.random.random(len(agent.gene)) < self.mutation_bitwise_rate
        new_agent = agent.__class__(
            [
                (agent.gene[i] if choice[i] else np.random.uniform())
                for i in range(len(choice))
            ],
            [
                (agent.param_gene[i] if choice[i] else agent.rules[i].generate_param())
                for i in range(len(choice))
            ],
        )
        logger.debug(
            f"Mutating agent's gene from '{agent.gene}' to '{new_agent.gene}', param_gene from '{agent.param_gene}' to '{new_agent.param_gene}' "
        )
        return new_agent


if __name__ == "__main__":
    population = [GeneticRealAgent() for _ in range(100)]
    RealEvolution(0.5, 0.5, 0.2, 0.3, 0.4).evolve(population, np.random.random(100))
