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

    def test_repr(self):
        print(self.c1)

    def test_update_accuracy(self):
        # with seed == 39, c1._strategy == 4
        actual = 1
        self.c1._update_accuracy(actual)
        self.assertEqual(self.c1.used, 1)
        self.assertEqual(self.c1.accuracy, self.c1._theta * (actual - self.c1._strategy) ** 2)


class TestPredictors(unittest.TestCase):

    def setUp(self):
        np.random.seed(39)
        self.p1 = Predictors(10, 16, 8, [0.1, 0.1, 0.8], 0.02, symm=True)

    def test_setUp(self):
        self.assertEqual(len(self.p1.predictors), 10)
        self.assertIn(Chromosome('2' * 16, '0' * 8, 0.02, symm=True), self.p1.predictors)
        self.assertFalse(self.p1.current)

    def test_find_winners1(self):
        for j in range(10):
            if j % 2:
                self.p1.predictors[j].used = j
                self.p1.predictors[j].accuracy = j / 100
        self.p1.find_winners(5)
        for i in self.p1.predictors:
            with self.subTest(i=i):
                self.assertGreater(i.accuracy, 0)
        
    def test_find_winners2(self):
        for j in range(10):
            if j % 2:
                self.p1.predictors[j].used = j
                self.p1.predictors[j].accuracy = j / 100
        self.p1.find_winners(3)
        for i in self.p1.predictors:
            with self.subTest(i=i):
                self.assertGreater(i.accuracy, 0.3)

    def test_match_state(self):
        '''
        With numpy random seed == 39, the Chromosomes are:
        0. Chromosome(2222222222222222, 00000000, 0.02)
        1. Chromosome(2221222222222222, 10100011, 0.02)
        2. Chromosome(0222222121221222, 11000110, 0.02)
        3. Chromosome(2222122022122222, 10100101, 0.02)
        4. Chromosome(2220222022222222, 11101011, 0.02)
        5. Chromosome(2221222222222202, 00011010, 0.02)
        6. Chromosome(2222220021221202, 10000111, 0.02)
        7. Chromosome(2122212222221221, 01000010, 0.02)
        8. Chromosome(2122222222222222, 00000011, 0.02)
        9. Chromosome(1222222222222222, 01100001, 0.02)

        match_state chooses the Chromosome that matches the state 
        and has the lowest accuracy

        If the state is all 1: 1111111111111111, then
        Chromosomes 0, 1, 7, 8, 9 all match
        test 1: Set all Chromosome accuracy to j/100, then match_state chooses Chromosome 0
        test 2: Set Chromosome 0 accuracy to 0.05, Chromosome 7 accuracy to 0.01, then
        match state chooses Chromosomes 1 and 7.
        '''
        state = '1111111111111111'
        c_len = 16
        for j in range(10):
            self.p1.predictors[j].used = 1
            self.p1.predictors[j].accuracy = j / 100
        self.p1.match_state(state, c_len)
        self.assertEqual(len(self.p1.current), 1)
        self.assertIn(Chromosome('2' * 16, '0' * 8, 0.02, symm=True), self.p1.current)
        self.p1.predictors[0].accuracy = 0.05
        self.p1.predictors[7].accuracy = 0.01
        self.p1.match_state(state, c_len)
        self.assertEqual(len(self.p1.current), 2)
        self.assertIn(Chromosome('2221222222222222', '10100011', 0.02, symm=True), self.p1.current)
        self.assertIn(Chromosome('2122212222221221', '01000010', 0.02, symm=True), self.p1.current)
        
        