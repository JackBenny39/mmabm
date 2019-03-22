import unittest

import numpy as np

from mmabm.genetics2 import Chromosome, Predictors


class TestChromosome(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(39)
        self.c1 = self._makeChromosome(''.join(str(x) for x in np.random.choice(np.arange(0, 3), 10, p=[0.05, 0.05, 0.9])),
                                       ''.join(str(x) for x in np.random.choice(np.arange(0, 2), 5)), 0.02, True)
        self.c2 = self._makeChromosome(''.join(str(x) for x in np.random.choice(np.arange(0, 3), 16, p=[0.05, 0.05, 0.9])),
                                       ''.join(str(x) for x in np.random.choice(np.arange(0, 2), 8)), 0.04, False)

    def _makeChromosome(self, condition, action, theta, symm):
        return Chromosome(condition, action, theta, symm)

    def test_setUp(self):
        self.assertEqual(len(self.c1.condition), 10)
        self.assertTrue(x==0 or x==1 or x==2 for x in self.c1.condition)
        self.assertEqual(len(self.c1.action), 5)
        self.assertTrue(x==0 or x==1 for x in self.c1.action)
        self.assertEqual(self.c1._strategy, int(self.c1.action[1:], 2)*(1 if int(self.c1.action[0]) else -1), 2)
        self.assertEqual(self.c1._theta, 0.02)
        self.assertFalse(self.c1.used)
        self.assertFalse(self.c1.accuracy)
        self.assertEqual(len(self.c2.condition), 16)
        self.assertTrue(x==0 or x==1 or x==2 for x in self.c2.condition)
        self.assertEqual(len(self.c2.action), 8)
        self.assertTrue(x==0 or x==1 for x in self.c2.action)
        self.assertEqual(self.c2._strategy, int(self.c2.action, 2))
        self.assertEqual(self.c2._theta, 0.04)
        self.assertFalse(self.c1.used)
        self.assertFalse(self.c1.accuracy)

    def test_update_accuracy(self):
        # with seed == 39, c1._strategy == 4
        self.c1._update_accuracy(1)
        self.assertEqual(self.c1.used, 1)
        self.assertEqual(self.c1.accuracy, 0.02 * (1 - 4) ** 2)