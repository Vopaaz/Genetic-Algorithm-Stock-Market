import sys

sys.path.append(".")

import pandas as pd
from experiment.background.agent import (
    GeneticAgent,
    GeneticSimpleAgent,
    GeneticRealAgent,
    GeneticBitAgent,
)


class Evolution(object):
    pass


class SimpleEvolution(Evolution):
    def __init__(self):
        pass


class BitEvolution(SimpleEvolution):
    pass


class RealEvolution(SimpleEvolution):
    pass


class ComplexEvolution(Evolution):
    pass


if __name__ == "__main__":
    pass
