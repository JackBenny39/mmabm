import random

import numpy as np

import mmabm.runner as runner


#h5_out = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\Trial %d\\ABMSmallCapSum.h5' % trial_no

settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025, 'qProvide': 0.5,
            'Taker': True, 'numTakers': 50, 'takerMaxQ': 1, 'tMu': 0.001,
            'InformedTrader': False, 'informedMaxQ': 1, 'informedRunLength': 1, 'iMu': 0.005,
            'PennyJumper': False, 'AlphaPJ': 0.05,
            'MarketMaker': True, 'NumMMs': 1, 'MMMaxQ': 1, 'MMQuotes': 12, 'MMQuoteRange': 60, 'MMDelta': 0.025,
            'QTake': True, 'WhiteNoise': 0.001, 'CLambda': 10.0, 'Lambda0': 100}

if __name__ == '__main__':
    
    seed = 51
    random.seed(seed)
    np.random.seed(seed)
    
    
    h5_root = 'python_profile_%d' % seed
    h5dir = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\Trial 1001\\'
    h5_file = '%s%s.h5' % (h5dir, h5_root)
        
    market1 = runner.Runner(h5filename=h5_file, **settings)
