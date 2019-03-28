import random
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
        self.assertEqual(self.c1.theta, 0.02)
        self.assertFalse(self.c1.used)
        self.assertFalse(self.c1.accuracy)
        self.assertEqual(len(self.c2.condition), 16)
        self.assertTrue(x==0 or x==1 or x==2 for x in self.c2.condition)
        self.assertEqual(len(self.c2.action), 8)
        self.assertTrue(x==0 or x==1 for x in self.c2.action)
        self.assertEqual(self.c2._strategy, int(self.c2.action, 2))
        self.assertEqual(self.c2.theta, 0.04)
        self.assertFalse(self.c1.used)
        self.assertFalse(self.c1.accuracy)

    def test_repr(self):
        print(self.c1)

    def test_update_accuracy(self):
        # with seed == 39, c1._strategy == 4
        actual = 1
        self.c1._update_accuracy(actual)
        self.assertEqual(self.c1.used, 1)
        self.assertEqual(self.c1.accuracy, self.c1.theta * (actual - self.c1._strategy) ** 2)


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
        
    def test_mutate(self):
        '''
        With numpy random seed == 39, random_sample((2, 10)) yields:
        [[0.54688916 0.79789902 0.82040188 0.12204987 0.60200201 
        0.52551458 0.46390841 0.47144574 0.63271284 0.92566388]
        [0.81550292 0.94444175 0.91958514 0.4148308  0.82581196 
        0.94626012 0.62804605 0.25226979 0.5923509  0.86040381]]
        For 0 or 1:
        If p_mutate == .42, then position 3 will be selected for the first string
        and positions 3 and 7 selected for the second string
        With random seed == 39, string 1 position 3 is 0, 
        string 2 position 3 is 1, position 7 is 1
        For 0, 1 or 2:
        If p_mutate == .42, then position 3 will be selected for the first string
        and positions 3 and 7 selected for the second string
        With random seed == 39, string 1 position 3 is 0, 
        string 2 position 3 is 1, position 7 is 1
        '''
        str1 = '1111111111'
        str2 = '0000000000'
        str_rng = 2
        str_len = len(str2)
        p_mutate = 0.42
        np.random.seed(39)
        random.seed(39)
        c1, c2 = self.p1.mutate(str1, str2, str_len, p_mutate, str_rng)
        self.assertEqual(c1, '1110111111')
        self.assertEqual(c2, '0001000100')
        str3 = '0122120120'
        str4 = '2102102102'
        str_rng = 3
        str_len = len(str3)
        p_mutate = 0.42
        np.random.seed(39)
        random.seed(39)
        c3, c4 = self.p1.mutate(str3, str4, str_len, p_mutate, str_rng)
        self.assertEqual(c3, '0120120120')
        self.assertEqual(c4, '2101102102')

    def test_cross(self):
        '''
        With random seed == 39, crossover occurs between 2 & 3
        '''
        str1 = '1111111111'
        str2 = '0000000000'
        str_len = len(str1)
        random.seed(39)
        c1, c2 = self.p1.cross(str1, str2, str_len)
        self.assertEqual(c1, '1110000000')
        self.assertEqual(c2, '0001111111')

    def test_check_chrom(self):
        pred_var = 1
        parent_var = 2
        # Conditions not equal -> use pred_var and append
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p = Chromosome('2222222222', '00000', 0.02, symm=True)
        c = Chromosome('2222022220', '00001', 0.02, symm=True)
        self.p1.check_chrom(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, pred_var)
        # Conditions equal, actions not equal -> use parent_var and append
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p.condition = '2222022220'
        p.action = '00000'
        self.p1.check_chrom(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, parent_var)
        # Conditions equal, actions equal -> no append
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p.condition = '2222022220'
        p.action = '00001'
        self.p1.check_chrom(c, p, pred_var, parent_var)
        self.assertFalse(self.p1.predictors)

    def test_check_chrom2(self):
        pred_var = 1
        parent_var = 2
        # Conditions not equal -> use pred_var and append
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p = Chromosome('2222222222', '00000', 0.02, symm=True)
        c = Chromosome('2222022220', '00001', 0.02, symm=True)
        self.p1.check_chrom2(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, pred_var)
        # Conditions equal, actions not equal -> use parent_var and append
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p.condition = '2222022220'
        p.action = '00000'
        self.p1.check_chrom2(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, parent_var)
        # Conditions equal, actions equal -> use parent_var, append anyway
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p.condition = '2222022220'
        p.action = '00001'
        self.p1.check_chrom2(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, parent_var)

    def test_new_genes_uf(self):
        '''
        Trim predictors to 5
        with seeds set to 39, Chromosomes 1 and 2 are selected 
        and 2 suitable children result
        '''
        random.seed(39)
        np.random.seed(39)
        p_len = 7
        a_len = len(self.p1.predictors[0].action)
        a_mutate = 0.05
        c_cross = 0.3
        c_len = len(self.p1.predictors[0].condition)
        c_mutate = 0.05
        self.p1.predictors = self.p1.predictors[:5]
        self.assertEqual(len(self.p1.predictors), 5)
        self.p1.new_genes_uf(p_len, a_len, a_mutate, c_cross, c_len, c_mutate)
        self.assertEqual(len(self.p1.predictors), 7)
        self.assertEqual(self.p1.predictors[5], Chromosome('2221122121221222', '10100010', 0.02, True))
        self.assertEqual(self.p1.predictors[6], Chromosome('0222222222222222', '11000111', 0.02, True))