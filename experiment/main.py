import sys

sys.path.append(".")

from experiment.background.agent import (
    GeneticBitAgent,
    GeneticRealAgent,
    GeneticComplexAgent,
)
from experiment.background.market import Market
from experiment.GA import BitEvolution, RealEvolution, ComplexEvolution
from experiment.util.config import TRAIN_START, TRAIN_END, TEST_START, TEST_END
from experiment import Experiment
from Lutil.checkpoints import checkpoint


def _experiment_suite(
    agent_class, evolution_class, result_prefix, n_population, evolution_params, epoch
):
    population = [agent_class() for _ in range(n_population)]
    evolution = evolution_class(*evolution_params)
    market = Market(TRAIN_START, TRAIN_END)
    test_market = Market(TEST_START, TEST_END)
    e = Experiment(population, evolution, market)
    e.train(epoch)
    e.test(test_market)
    e.visualize(
        f"./results/{result_prefix}-{n_population}-agent-param-{'-'.join([str(i) for i in evolution_params])}.png"
    )


@checkpoint
def real_experiment_suite(n_population, evolution_params, epoch):
    _experiment_suite(
        GeneticRealAgent, RealEvolution, "real", n_population, evolution_params, epoch
    )


@checkpoint
def bit_experiment_suite(n_population, evolution_params, epoch):
    _experiment_suite(
        GeneticBitAgent, BitEvolution, "bit", n_population, evolution_params, epoch
    )


@checkpoint
def complex_experiment_suite(n_population, evolution_params, epoch):
    _experiment_suite(
        GeneticComplexAgent,
        ComplexEvolution,
        "complex",
        n_population,
        evolution_params,
        epoch,
    )


if __name__ == "__main__":

    bit_experiment_suite(20, [0.6, 0.75, 0.1], 10)
    bit_experiment_suite(20, [0.5, 0.75, 0.1], 10)
    bit_experiment_suite(20, [0.4, 0.75, 0.1], 10)

    bit_experiment_suite(20, [0.6, 0.5, 0.1], 10)
    bit_experiment_suite(20, [0.6, 0.25, 0.1], 10)

    bit_experiment_suite(20, [0.6, 0.75, 0.2], 10)
    bit_experiment_suite(20, [0.6, 0.75, 0.3], 10)

    bit_experiment_suite(20, [0.6, 0.75, 0.1, 0.1], 10)
    bit_experiment_suite(20, [0.6, 0.75, 0.1, 0.2], 10)

    real_experiment_suite(20, [0.6, 0.75, 0.1], 10)
    real_experiment_suite(20, [0.5, 0.75, 0.1], 10)
    real_experiment_suite(20, [0.4, 0.75, 0.1], 10)

    real_experiment_suite(20, [0.6, 0.5, 0.1], 10)
    real_experiment_suite(20, [0.6, 0.25, 0.1], 10)

    real_experiment_suite(20, [0.6, 0.75, 0.2], 10)
    real_experiment_suite(20, [0.6, 0.75, 0.3], 10)

    real_experiment_suite(20, [0.6, 0.75, 0.1, 0.1], 10)
    real_experiment_suite(20, [0.6, 0.75, 0.1, 0.2], 10)

    complex_experiment_suite(20, [0.6, 0.75, 0.1], 10)
    complex_experiment_suite(20, [0.5, 0.75, 0.1], 10)
    complex_experiment_suite(20, [0.4, 0.75, 0.1], 10)

    complex_experiment_suite(20, [0.6, 0.5, 0.1], 10)
    complex_experiment_suite(20, [0.6, 0.25, 0.1], 10)

    complex_experiment_suite(20, [0.6, 0.75, 0.2], 10)
    complex_experiment_suite(20, [0.6, 0.75, 0.3], 10)

    complex_experiment_suite(20, [0.6, 0.75, 0.1, 0.1], 10)
    complex_experiment_suite(20, [0.6, 0.75, 0.1, 0.2], 10)

