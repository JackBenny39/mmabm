import random
import time

import numpy as np

import mmabm.runnerc as runnerc


#h5_out = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\Trial %d\\ABMSmallCapSum.h5' % trial_no

settings = {'Provider': True, 'numProviders': 38, 'providerMaxQ': 1, 'pAlpha': 0.0375, 'pDelta': 0.025, 'qProvide': 0.5,
            'Taker': True, 'numTakers': 50, 'takerMaxQ': 1, 'tMu': 0.001,
            'InformedTrader': False, 'informedMaxQ': 1, 'informedRunLength': 1, 'iMu': 0.005,
            'PennyJumper': False, 'AlphaPJ': 0.05,
            'MarketMaker': True, 'NumMMs': 1, 'MMMaxQ': 1, 'MMQuotes': 12, 'MMQuoteRange': 60, 'MMDelta': 0.025,
            'QTake': True, 'WhiteNoise': 0.001, 'CLambda': 10.0, 'Lambda0': 100}

if __name__ == '__main__':
    
    print(time.time())
    
    for j in range(51, 61):
        random.seed(j)
        np.random.seed(j)
    
        start = time.time()
    
        h5_root = 'cython_super_%d' % j
        h5dir = 'C:\\Users\\user\\Documents\\Agent-Based Models\\h5 files\\Trial 1001\\'
        h5_file = '%s%s.h5' % (h5dir, h5_root)
        
        market1 = runnerc.Runner(h5filename=h5_file, **settings)

        print('Run %d: %.1f seconds' % (j, time.time() - start))
