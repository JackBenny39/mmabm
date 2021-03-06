import random

import numpy as np


def make_chromosome(num_genes, condition_len, action_len, condition_probs):
    genes = {'2' * condition_len: '0' * action_len}
    while len(genes) < num_genes:
        gk = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), condition_len, p=condition_probs))
        gv = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), action_len))
        genes.update({gk: gv})
    return genes

def find_winners(cromosome, c_len, meas, keep, rev=True):
    allk = '2' * c_len
    alld = {allk: cromosome.pop(allk)}
    chrom = dict(sorted(cromosome.items(), key=lambda kv: kv[1][meas][-1], reverse=rev)[:keep - 1])
    chrom.update(alld)
    return chrom

def make_strat(chroms, meas, maxi=True, symm=True):
    fit = 1000 if maxi else 0
    if symm:
        strat = {k: {'action': v, 'strategy': int(v[1:], 2)*(1 if int(v[0]) else -1), meas: [0, 0, fit]} for k, v in chroms.items()}
    else:
        strat =  {k: {'action': v, 'strategy': int(v, 2), meas: [0, 0, fit]} for k, v in chroms.items()}
    return strat, len(list(chroms.keys())[0]), len(strat)

def make_weights(l):
    ranger = [j for j in range(1, l+1)]
    denom = sum(ranger)
    numer = reversed(ranger)
    return np.cumsum([k/denom for k in numer])

def match_strat_random(market_state, meas, strat, c_len):
    '''Returns a randomly chosen strategy from all strategies with the maximum accuracy'''
    temp_strats = []
    max_strength = 0
    max_accuracy = 0
    for cond in strat.keys():
        if all([(cond[x] == market_state[x] or cond[x] == '2') for x in range(c_len)]):
            strength = sum([cond[x] == market_state[x] for x in range(c_len)])
            if strength > max_strength:
                temp_strats.clear()
                temp_strats.append(cond)
                max_strength = strength
                max_accuracy = strat[cond][meas][-1]
            elif strength == max_strength:
                if strat[cond][meas][-1] > max_accuracy:
                    temp_strats.clear()
                    temp_strats.append(cond)
                    max_accuracy = strat[cond][meas][-1]
                elif strat[cond][meas][-1] == max_accuracy:
                    temp_strats.append(cond)
    return random.choice(temp_strats)

def match_strat_all(state, meas, strat, c_len):
    '''Returns all strategies with the maximum accuracy'''
    current_strat = []
    max_strength = 0
    max_rs = 0
    for cond in strat.keys():
        if all([(cond[x] == state[x] or cond[x] == '2') for x in range(c_len)]):
            strength = sum([cond[x] == state[x] for x in range(c_len)])
            if strength > max_strength:
                current_strat.clear()
                current_strat.append(cond)
                max_strength = strength
                max_rs = strat[cond][meas][-1]
            elif strength == max_strength:
                if strat[cond][meas][-1] > max_rs:
                    current_strat.clear()
                    current_strat.append(cond)
                    max_rs = strat[cond][meas][-1]
                elif strat[cond][meas][-1] == max_rs:
                    current_strat.append(cond)
    return current_strat

def new_genes_wf(cromosome, gene_num, weights, c_len, mutate_p, a_len, meas, m_func, maxi=True):
    # Step 1: get the genes
    parents = list(cromosome.keys())
    # Step 2: update the strategy dict with unique new children
    while len(cromosome) < gene_num:
        # Choose two parents - uniform selection
        p1, p2 = tuple(random.choices(parents, cum_weights=weights, k=2))
        # Random uniform crossover
        x = random.randrange(c_len)
        p = p1[:x] + p2[x:]
        # Random uniform mutate
        if random.random() < mutate_p:
            z = random.randrange(c_len)
            p = p[:z] + str(random.randrange(3)) + p[z+1:]
        # Check if new child differs from current parents
        if p not in list(cromosome.keys()):
            # Update child action & strategy
            y = random.random()
            if y < 0.333: # choose parent 1
                a = cromosome[p1]['action']
                s = cromosome[p1]['strategy']
            elif y > 0.667: # choose parent 2
                a = cromosome[p2]['action']
                s = cromosome[p2]['strategy']
            else: # average parent 1 & 2
                s = int((cromosome[p1]['strategy'] + cromosome[p2]['strategy']) / 2)
                a = m_func(s, a_len)
            # Update accuracy - weighted average
            a0 = cromosome[p1][meas][0] + cromosome[p2][meas][0]
            a1 = cromosome[p1][meas][1] + cromosome[p2][meas][1]
            a2 = a0/a1 if a1 > 0 else 0
            a2 = 1000 - a2 if maxi else a2
            accuracy = [a0, a1, a2]
            # Add new child to strategy dict
            cromosome.update({p: {'action': a, 'strategy': s, meas: accuracy}})
    return cromosome

def new_genes_uf(cromosome, gene_num, c_len, mutate_p, a_len, meas, m_func, maxi=True):
    # Step 1: get the genes
    parents = list(cromosome.keys())
    # Step 2: update the strategy dict with unique new children
    while len(cromosome) < gene_num:
        # Choose two parents - uniform selection
        p1, p2 = tuple(random.sample(parents, 2))
        # Random uniform crossover
        x = random.randrange(c_len)
        p = p1[:x] + p2[x:]
        # Random uniform mutate
        if random.random() < mutate_p:
            z = random.randrange(c_len)
            p = p[:z] + str(random.randrange(3)) + p[z+1:]
        # Check if new child differs from current parents
        if p not in list(cromosome.keys()):
            # Update child action & strategy
            y = random.random()
            if y < 0.333: # choose parent 1
                a = cromosome[p1]['action']
                s = cromosome[p1]['strategy']
            elif y > 0.667: # choose parent 2
                a = cromosome[p2]['action']
                s = cromosome[p2]['strategy']
            else: # average parent 1 & 2
                s = int((cromosome[p1]['strategy'] + cromosome[p2]['strategy']) / 2)
                a = m_func(s, a_len)
            # Update accuracy - weighted average
            a0 = cromosome[p1][meas][0] + cromosome[p2][meas][0]
            a1 = cromosome[p1][meas][1] + cromosome[p2][meas][1]
            a2 = a0/a1 if a1 > 0 else 0
            a2 = 1000 - a2 if maxi else a2
            accuracy = [a0, a1, a2]
            # Add new child to strategy dict
            cromosome.update({p: {'action': a, 'strategy': s, meas: accuracy}})
    return cromosome

