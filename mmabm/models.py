from mmabm.genetics2 import Predictors

class Model:
    '''
    A model defines the predictors and a function to convert the chromosome
    action to a market structure variable.
    '''

    def __init__(self, num_chroms, condition_len, action_len, condition_probs, 
                 action_mutate_p, condition_cross_p, condition_mutate_p, 
                 theta, keep_pct, symm, weights):
        self.predictors = Predictors(num_chroms, condition_len, action_len, condition_probs, 
                 action_mutate_p, condition_cross_p, condition_mutate_p, 
                 theta, keep_pct, symm, weights)



def order_imbalance():
    '''
    order imbalance: 24 bits, 12 for previous period and 12 for previous 5 periods:
    previous period -> one bit each for < -8, -4, -3, -2, -1, 0 and > 0, 1, 2, 3, 4, 8
    previous 5 periods -> one bit each for < -16, -8, -6, -4, -2, 0 and > 0, 2, 4, 6, 8, 16

    The market maker has a set of predictors (condition/forecast rules) where the condition
    matches the market descriptors (i.e., the market state) and the forecasts are used as inputs
    to the market maker decision making.

    Each predictor condition is a bit string that coincides with market descriptors with the
    additional possibility of "don't care" (==2). 
    Each predictor condition has an associated forecast
    order imbalance: 6 bits -> lhs bit is +1/-1 and 2^5 - 1 = 31 for a range of -31 - +31 (symm==True)

    Example:
    order imbalance signal: 011111000000011111000000
    < -4 for previous period and < -8 for previous 5 periods
    order imbalance gene: 222221022222222122222012: 010010
    this gene does not match the market state in position 23 and forecasts an order
    imbalance of +18 (+1*(1*16 + 0*8 + 0*4 + 1*2 + 0*1))
    '''
    num_chroms = 100
    condition_len = 24
    action_len = 6
    condition_probs = [0.05, 0.05, 0.9]
    action_mutate_p = 0.06
    condition_cross_p = 0.1
    condition_mutate_p = 0.06
    theta = 0.02
    keep_pct = 0.8
    symm = True
    weights = False

    return Predictors(num_chroms, condition_len, action_len, condition_probs, 
                      action_mutate_p, condition_cross_p, condition_mutate_p, 
                      theta, keep_pct, symm, weights)

def update_midpoint():
    pass
