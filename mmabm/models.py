import numpy as np

def make_chromosome(num_genes, condition_len, action_len, condition_probs):
    genes = {'2' * condition_len: '0' * action_len}
    while len(genes) < num_genes:
        gk = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), condition_len, p=condition_probs))
        gv = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), action_len))
        genes.update({gk: gv})
    return genes
