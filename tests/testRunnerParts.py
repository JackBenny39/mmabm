import random
import unittest

from mmabm.trader import InformedTrader


class TestTrader(unittest.TestCase):
    
    def buildInformedTrader2(self, informedMaxQ, informedRunLength, informedTrades, prime1):
        ''' Informed trader id starts with 5
        '''
        informed = InformedTrader(5000, informedMaxQ)
        numChoices = int(informedTrades/(informedRunLength*informed.quantity)) + 1
        choiceRange = range(prime1, 100000 - informedRunLength + 1)
        t_delta_i = set()
        for _ in range(1, numChoices):
            runL = 0
            step = random.choice(choiceRange)
            while runL < informedRunLength:
                while step in t_delta_i:
                    step += 1
                t_delta_i.add(step)
                step += 1
                runL += 1
        return t_delta_i, informed
    
    def test_buildInformedTrader2(self):
        t_delta_i, informed_trader = self.buildInformedTrader2(1, 2, 50, 20)
        sorted_t = list(t_delta_i)
        sorted_t.sort()
        print(len(sorted_t), " : ", sorted_t, informed_trader)