import random

import numpy as np

from operator import attrgetter


class Chromosome:
    '''
    Chromosome is a collection of chromosome attributes:

    condition is a bitstring of 0, 1, or 2; it matches the market state
    action is a bitstring of 0 or 1; it is the action associated with the market state
    strategy is the numerical value of _action
    used is 1 if used, 0 o/w 
    accuracy is the MA of the MSFE
    theta is the MA weight

    Each of the bits can be thought of as a gene - and are subject to potential mutation
    A chromosome is a collection of genes - and is subject to potential crossover with another chromosome
    '''

    def __init__(self, condition, action, theta, symm):
        self.condition = condition #''.join(str(x) for x in np.random.choice(np.arange(0, 3), condition_len, p=condition_probs))
        self.action = action # ''.join(str(x) for x in np.random.choice(np.arange(0, 2), action_len))
        self.symm = symm
        self.strategy = self._convert_action() #int(self.action[1:], 2)*(1 if int(self.action[0]) else -1) if symm else int(self.action, 2)
        self.used = 0
        self.accuracy = 0
        self.theta = theta

    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3}, {4})'.format(class_name, self.condition, self.action, self.theta, self.symm)

    def __str__(self):
        return str(tuple([self.condition, self.action, self.strategy, self.used, self.accuracy, self.theta, self.symm]))

    def __eq__(self, other):
        return self.condition == other.condition and self.action == other.action

    def _convert_action(self):
        return int(self.action[1:], 2)*(1 if int(self.action[0]) else -1) if self.symm else int(self.action, 2)

    def update_accuracy(self, actual):
        self.used = 1
        self.accuracy = (1 - self.theta) * self.accuracy + self.theta * (actual - self.strategy) ** 2


class Predictors:
    '''
    Predictors is a list of all chromosomes and a list of currently active chromosomes
    
    '''
    
    def __init__(self, num_chroms, condition_len, action_len, condition_probs, 
                 action_mutate_p, condition_cross_p, condition_mutate_p, 
                 theta, keep_pct, symm, weights):
        self._num_chroms = num_chroms
        self._condition_len = condition_len
        self._action_len = action_len
        self._action_mutate_p = action_mutate_p
        self._condition_cross_p = condition_cross_p
        self._condition_mutate_p = condition_mutate_p
        self.predictors = [Chromosome('2' * condition_len, '0' * action_len, theta, symm)]
        self._make_predictors(condition_probs, theta, symm)
        self._keep = int(keep_pct * num_chroms)
        if not weights:
            self.new_genes = self._new_genes_uf
        else:
            self._weights = self._make_weights()
            self.new_genes = self._new_genes_wf
        self.current = []

    def _make_predictors(self, condition_probs, theta, symm):
        while len(self.predictors) < self._num_chroms:
            c = Chromosome(''.join(str(x) for x in np.random.choice(np.arange(0, 3), self._condition_len, p=condition_probs)),
                           ''.join(str(x) for x in np.random.choice(np.arange(0, 2), self._action_len)), theta, symm)
            if c not in self.predictors:
                self.predictors.append(c)

    def _make_weights(self):
        ranger = [j for j in range(1, self._keep + 1)]
        denom = sum(ranger)
        numer = reversed(ranger)
        return np.cumsum([k/denom for k in numer])
    
    def _match_state(self, state):
        self.current.clear()
        min_acc = max([c.accuracy for c in self.predictors])
        for c in self.predictors:
            if all([(c.condition[x] == state[x] or c.condition[x] == '2') for x in range(self._condition_len)]):
                if c.accuracy < min_acc:
                    self.current.clear()
                    self.current.append(c)
                    min_acc = c.accuracy
                elif c.accuracy == min_acc:
                    self.current.append(c)

    def get_forecast(self, state):
        self._match_state(state)
        return sum([c.strategy for c in self.current]) / len(self.current)

    def update_accuracies(self, actual):
        for c in self.current:
            c.update_accuracy(actual)
    
    def _new_genes_uf(self):
        self._find_winners_uf()
        pred_var = np.mean([p.accuracy for p in self.predictors]) # if p.used?
        while len(self.predictors) < self._num_chroms:
            # Choose two parents - uniform selection
            p1, p2 = tuple(random.sample(self.predictors, 2))
            parent_var = (p1.accuracy + p2.accuracy) / 2
            # Random uniform crossover for action
            c1_action, c2_action = self._cross(p1.action, p2.action, self._action_len)
            # Random mutation with p = a_mutate for each gene (bit) in action
            c1_action, c2_action = self._mutate(c1_action, c2_action, self._action_len, self._action_mutate_p, 2)
            # Random uniform crossover for condition with p = c_cross
            if random.random() < self._condition_cross_p:
                c1_condition, c2_condition = self._cross(p1.condition, p2.condition, self._condition_len)
            else:
                c1_condition = p1.condition
                c2_condition = p2.condition
            # Random mutation with p = c_mutate for each gene (bit) in condition
            c1_condition, c2_condition = self._mutate(c1_condition, c2_condition, self._condition_len, self._condition_mutate_p, 3)
            # Make the children Chromosomes and check for uniqueness
            self._check_chrom(Chromosome(c1_condition, c1_action, p1.theta, p1.symm), p1, pred_var, parent_var)
            self._check_chrom(Chromosome(c2_condition, c2_action, p2.theta, p2.symm), p2, pred_var, parent_var)

    def _new_genes_wf(self):
        self._find_winners_wf()
        pred_var = np.mean([p.accuracy for p in self.predictors]) # if p.used?
        while len(self.predictors) < self._num_chroms:
            # Choose two parents - weighted selection
            p1, p2 = tuple(random.choices(self.predictors, cum_weights=self._weights, k=2))
            parent_var = (p1.accuracy + p2.accuracy) / 2
            # Random uniform crossover for action
            c1_action, c2_action = self._cross(p1.action, p2.action, self._action_len)
            # Random mutation with p = a_mutate for each gene (bit) in action
            c1_action, c2_action = self._mutate(c1_action, c2_action, self._action_len, self._action_mutate_p, 2)
            # Random uniform crossover for condition with p = c_cross
            if random.random() < self._condition_cross_p:
                c1_condition, c2_condition = self._cross(p1.condition, p2.condition, self._condition_len)
            else:
                c1_condition = p1.condition
                c2_condition = p2.condition
            # Random mutation with p = c_mutate for each gene (bit) in condition
            c1_condition, c2_condition = self._mutate(c1_condition, c2_condition, self._condition_len, self._condition_mutate_p, 3)
            # Make the children Chromosomes and check for uniqueness
            self._check_chrom(Chromosome(c1_condition, c1_action, p1.theta, p1.symm), p1, pred_var, parent_var)
            self._check_chrom(Chromosome(c2_condition, c2_action, p2.theta, p2.symm), p2, pred_var, parent_var)

    def _find_winners_uf(self):
        used = [c for c in self.predictors if c.used]
        if len(used) < self._keep:
            self.predictors = sorted(self.predictors, key=attrgetter('used'), reverse=True)[: self._keep]
        else:
            self.predictors = sorted(used, key=attrgetter('accuracy'))[: self._keep]

    def _find_winners_wf(self):
        used = [c for c in self.predictors if c.used]
        if len(used) < self._keep:
            temp = sorted(self.predictors, key=attrgetter('accuracy'))
            self.predictors = sorted(temp, key=attrgetter('used'), reverse=True)[: self._keep]
        else:
            self.predictors = sorted(used, key=attrgetter('accuracy'))[: self._keep]
    
    def _mutate(self, str1, str2, str_len, mutate_prob, str_rng):
        m = np.random.random_sample((2, str_len))
        for j in range(str_len):
            if m[0, j] < mutate_prob:
                str1 = str1[:j] + str(random.randrange(str_rng)) + str1[j+1:]
            if m[1, j] < mutate_prob:
                str2 = str2[:j] + str(random.randrange(str_rng)) + str2[j+1:]
        return str1, str2

    def _cross(self, str1, str2, str_len):
        x = random.randrange(str_len)
        child1 = str1[:x] + str2[x:]
        child2 = str2[:x] + str1[x:]
        return child1, child2

    def _check_chrom(self, c, p, pred_var, parent_var):
        '''
        If condition and action are the same, don't add the child
        '''
        if c.condition != p.condition:
            c.accuracy = pred_var
            self.predictors.append(c)
        elif c.action != p.action:
            c.accuracy = parent_var
            self.predictors.append(c)

    def _check_chrom2(self, c, p, pred_var, parent_var):
        '''
        If condition and action are the same, add the child
        '''
        c.accuracy = pred_var if c.condition != p.condition else parent_var
        self.predictors.append(c)
