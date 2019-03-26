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
        self.symm = symm
        self._strategy = self._convert_action() #int(self.action[1:], 2)*(1 if int(self.action[0]) else -1) if symm else int(self.action, 2)
        self.used = 0
        self.accuracy = 0
        self.theta = theta

    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3}, {4})'.format(class_name, self.condition, self.action, self.theta, self.symm)

    def __str__(self):
        return str(tuple([self.condition, self.action, self._strategy, self.used, self.accuracy, self.theta, self.symm]))

    def __eq__(self, other):
        return self.condition == other.condition and self.action == other.action

    def _convert_action(self):
        return int(self.action[1:], 2)*(1 if int(self.action[0]) else -1) if self.symm else int(self.action, 2)

    def _update_accuracy(self, actual):
        self.used = 1
        self.accuracy = (1 - self.theta) * self.accuracy + self.theta * (actual - self._strategy) ** 2


class Predictors:
    '''
    Predictors is a list of all chromosomes and a list of currently active chromosomes
    
    '''
    
    def __init__(self, num_chroms, condition_len, action_len, condition_probs, theta, symm=True):
        self.predictors = [Chromosome('2' * condition_len, '0' * action_len, theta, symm)]
        self._make_predictors(num_chroms, condition_len, action_len, condition_probs, theta, symm)
        self.current = []

    def _make_predictors(self, num_chroms, condition_len, action_len, condition_probs, theta, symm):
        while len(self.predictors) < num_chroms:
            c = Chromosome(''.join(str(x) for x in np.random.choice(np.arange(0, 3), condition_len, p=condition_probs)),
                           ''.join(str(x) for x in np.random.choice(np.arange(0, 2), action_len)), theta, symm)
            if c not in self.predictors:
                self.predictors.append(c)
    
    def find_winners(self, keep):
        used = [c for c in self.predictors if c.used]
        if len(used) <= keep:
            self.predictors = sorted(self.predictors, key=lambda c: c.used, reverse=True)[: keep]
        else:
            self.predictors = sorted(used, key=lambda c: c.accuracy)[: keep]
    
    def match_state(self, state, c_len):
        self.current.clear()
        min_acc = max([c.accuracy for c in self.predictors])
        for c in self.predictors:
            if all([(c.condition[x] == state[x] or c.condition[x] == '2') for x in range(c_len)]):
                if c.accuracy < min_acc:
                    self.current.clear()
                    self.current.append(c)
                    min_acc = c.accuracy
                elif c.accuracy == min_acc:
                    self.current.append(c)
    
    def new_genes_uf(self, p_len, a_len, a_mutate, c_cross, c_len, c_mutate):
        pred_var = np.mean([p.accuracy for p in self.predictors])
        while len(self.predictors) < p_len:
            # Choose two parents - uniform selection
            p1, p2 = tuple(random.sample(self.predictors, 2))
            parent_var = (p1.accuracy + p2.accuracy) / 2
            # Random uniform crossover for action
            c1_action, c2_action = self.cross(p1.action, p2.action, a_len)
            # Random mutation with p = a_mutate for each gene (bit) in action
            c1_action, c2_action = self.mutate(c1_action, c2_action, a_len, a_mutate, 2)
            # Random uniform crossover for condition with p = c_cross
            if random.random() < c_cross:
                c1_condition, c2_condition = self.cross(p1.condition, condition, c_len)
            else:
                c1_condition = p1.condition
                c2_condition = p2.condition
            # Random mutation with p = c_mutate for each gene (bit) in condition
            c1_condition, c2_condition = self.mutate(c1_condition, c2_condition, c_len, c_mutate, 3)
            # Make the children Chromosomes and check for uniqueness
            self.check_chrom(Chromosome(c1_condition, c1_action, p1.theta, p1.symm), p1, pred_var, parent_var)
            self.check_chrom(Chromosome(c2_condition, c2_action, p2.theta, p2.symm), p2, pred_var, parent_var)

        def mutate(self, str1, str2, str_len, mutate_prob, str_rng):
            m = np.random.random_sample((2, str_len))
            for j in range(str_len):
                if m[0, j] < mutate_prob:
                    str1 = str1[:j] + str(random.randrange(str_rng)) + str1[j+1:]
                if m[1, j] < mutate_prob:
                    str2 = str2[:j] + str(random.randrange(str_rng)) + str2[j+1:]
            return str1, str2

        def cross(self, str1, str2, str_len):
            x = random.randrange(str_len)
            child1 = str1[:x] + str2[x:]
            child2 = str2[:x] + str1[x:]
            return child1, child2

        def check_chrom(self, c, p, pred_var, parent_var):
            '''
            If condition and action are the same, don't add the child
            '''
            if c.condition != p.condition:
                c.accuracy = pred_var
                self.predictors.append(c)
            elif c.action != p.action:
                c.accuracy = parent_var
                self.predictors.append(c)

        def check_chrom2(self, c, p, pred_var, parent_var):
            '''
            If condition and action are the same, add the child
            '''
            c.accuracy = pred_var if c.condition != p.condition else parent_var
            self.predictors.append(c)