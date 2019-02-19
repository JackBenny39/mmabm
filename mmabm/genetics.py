import random

import numpy as np


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

