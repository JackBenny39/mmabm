import random

import numpy as np


class Chromosome:
    '''
    Chromosome is a collection of chromosome attributes:

    _condition is a bitstring of 0, 1, or 2; it matches the market state
    _action is a bitstring of 0 or 1; it is the action associated with the market state
    _strategy is the numerical value of _action
    _accuracy is a list of three items: a usage count, the MA of the MSFE, and it's inverse
    _theta is the MA weight

    Each of the bits can be thought of as a gene - and are subject to potential mutation
    A chromosome is a collection of genes - and is subject to potential crossover with another chromosome
    '''

    def __init__(self, condition_len, action_len, condition_probs, theta, symm):
        self._condition = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), condition_len, p=condition_probs))
        self._action = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), action_len))
        self._strategy = int(self._action[1:], 2)*(1 if int(self._action[0]) else -1) if symm else int(self._action, 2)
        self._accuracy = [0, 0, 0]
        self._theta = theta

    def _update_accuracy(self, actual):
        self._accuracy[0] += 1
        self._accuracy[1] = (1 - self._theta) * self._accuracy[1] + self._theta * (actual - self._strategy) ** 2
        self._accuracy[2] = 1 / self._accuracy[1] if self._accuracy[1] > 0 else 0


def make_predictors(num_chroms, condition_len, action_len, condition_probs, theta, symm=True):
    predictors = []
    while len(predictors) < num_chroms:
        c = Chromosome(condition_len, action_len, condition_probs, theta, symm)
        if c not in predictors:
            predictors.append(c)
    return predictors