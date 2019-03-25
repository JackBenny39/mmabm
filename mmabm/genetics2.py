import random

import numpy as np


class Chromosome:
    '''
    Chromosome is a collection of chromosome attributes:

    condition is a bitstring of 0, 1, or 2; it matches the market state
    action is a bitstring of 0 or 1; it is the action associated with the market state
    _strategy is the numerical value of _action
    used is the number of events 
    accuracy is the MA of the MSFE
    _theta is the MA weight

    Each of the bits can be thought of as a gene - and are subject to potential mutation
    A chromosome is a collection of genes - and is subject to potential crossover with another chromosome
    '''

    def __init__(self, condition, action, theta, symm):
        self.condition = condition #''.join(str(x) for x in np.random.choice(np.arange(0, 3), condition_len, p=condition_probs))
        self.action = action # ''.join(str(x) for x in np.random.choice(np.arange(0, 2), action_len))
        self._strategy = self._convert_action(symm) #int(self.action[1:], 2)*(1 if int(self.action[0]) else -1) if symm else int(self.action, 2)
        self.used = 0
        self.accuracy = 0
        self._theta = theta

    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3}, {4}, {5}, {6})'.format(class_name, self.condition, self.action, self._strategy,
                                                          self.used, self.accuracy, self._theta)

    def __eq__(self, other):
        return self.condition == other.condition and self.action == other.action

    def _convert_action(self, symm):
        return int(self.action[1:], 2)*(1 if int(self.action[0]) else -1) if symm else int(self.action, 2)

    def _update_accuracy(self, actual):
        self.used = 1
        self.accuracy = (1 - self._theta) * self.accuracy + self._theta * (actual - self._strategy) ** 2


class Predictors:
    '''
    Predictors is a list of chromosomes
    
    '''
    
    def __init__(self, num_chroms, condition_len, action_len, condition_probs, theta, symm=True):
        self.predictors = [Chromosome('2' * condition_len, '0' * action_len, theta, symm)]
        self._make_predictors(num_chroms, condition_len, action_len, condition_probs, theta, symm)

    def _make_predictors(self, num_chroms, condition_len, action_len, condition_probs, theta, symm):
        while len(self.predictors) < num_chroms:
            c = Chromosome(''.join(str(x) for x in np.random.choice(np.arange(0, 3), condition_len, p=condition_probs)),
                           ''.join(str(x) for x in np.random.choice(np.arange(0, 2), action_len)), theta, symm)
            if c not in self.predictors:
                self.predictors.append(c)
    
    def find_winners(self, keep):
        used = [c for c in self.predictors if c.used]
        if len(used) <= keep:
            self.predictors = sorted(self.predictors, key=lambda c: c.used, reverse=True)[: keep - 1]
        else:
            self.predictors = sorted(used, key=lambda c: c.accuracy)[: keep - 1]
    
    def match_state(self, state, c_len):
        current_preds = []
        min_acc = max([c.accuracy for c in self.predictors])
        for c in self.predictors:
            if all([(c.condition[x] == state[x] or c.condition[x] == '2') for x in range(c_len)]):
                if c.accuracy < min_acc:
                    current_preds.clear()
                    current_preds.append(c)
                    min_acc = c.accuracy
                elif c.accuracy == min_acc:
                    current_preds.append(c)
        return current_preds
    
    def new_genes_uf(self, p_len, a_len):
        while len(self.predictors) < p_len:
            # Choose two parents - uniform selection
            p1, p2 = tuple(random.sample(self.predictors, 2))
            # Random uniform crossover for action
            x = random.randrange(a_len)
            c1_action = p1.action[:x] + p2.action[x:]
            c2_action = p2.action[:x] + p1.action[x:]