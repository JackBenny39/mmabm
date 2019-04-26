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
        self.assertTrue(x in '012' for x in self.c1.condition)
        self.assertEqual(len(self.c1.action), 5)
        self.assertTrue(x in '01' for x in self.c1.action)
        # test _convert_action
        self.assertEqual(self.c1.strategy, int(self.c1.action[1:], 2)*(1 if int(self.c1.action[0]) else -1), 2)
        self.assertEqual(self.c1.theta, 0.02)
        self.assertFalse(self.c1.used)
        self.assertFalse(self.c1.accuracy)
        self.assertEqual(len(self.c2.condition), 16)
        self.assertTrue(x in '012' for x in self.c2.condition)
        self.assertEqual(len(self.c2.action), 8)
        self.assertTrue(x in '01' for x in self.c2.action)
        # test _convert_action
        self.assertEqual(self.c2.strategy, int(self.c2.action, 2))
        self.assertEqual(self.c2.theta, 0.04)
        self.assertFalse(self.c1.used)
        self.assertFalse(self.c1.accuracy)

    def test_repr(self):
        self.assertEqual(self.c1.__repr__(), 'Chromosome(2222222222, 10100, 0.02, True)')

    def test_str(self):
        self.assertEqual(str(self.c1), "('2222222222', '10100', 4, 0, 0, 0.02, True)")

    def test_eq(self):
        self.c3 = self._makeChromosome(''.join(str(x) for x in np.random.choice(np.arange(0, 3), 10, p=[0.05, 0.05, 0.9])),
                                       ''.join(str(x) for x in np.random.choice(np.arange(0, 2), 5)), 0.02, True)
        self.c3.condition = self.c2.condition
        self.c3.action = self.c2.action
        self.assertEqual(self.c3, self.c2)
        self.c3.action = self.c1.action
        self.assertNotEqual(self.c3, self.c2)
        self.c3.condition = self.c1.condition
        self.c3.action = self.c2.action
        self.assertNotEqual(self.c3, self.c2)

    def test_update_accuracy(self):
        # with seed == 39, c1._strategy == 4
        actual = 1
        self.c1.update_accuracy(actual)
        self.assertEqual(self.c1.used, 1)
        self.assertEqual(self.c1.accuracy, self.c1.theta * (actual - self.c1.strategy) ** 2)


class TestPredictors(unittest.TestCase):

    def setUp(self):
        np.random.seed(39)
        num_chroms = 10
        condition_len = 16
        action_len = 8
        condition_probs = [0.1, 0.1, 0.8]
        action_mutate_p = 0.42
        condition_cross_p = 0.3
        condition_mutate_p = 0.42
        theta = 0.02
        keep_pct = 0.5
        symm=True
        weights = False
        self.p1 = Predictors(num_chroms, condition_len, action_len, condition_probs, 
                             action_mutate_p, condition_cross_p, condition_mutate_p, 
                             theta, keep_pct, symm, weights)
        weights = True
        self.p2 = Predictors(num_chroms, condition_len, action_len, condition_probs, 
                             action_mutate_p, condition_cross_p, condition_mutate_p, 
                             theta, keep_pct, symm, weights)

    def test_setUp(self):
        self.assertEqual(self.p1._num_chroms, 10)
        self.assertEqual(self.p1._condition_len, 16)
        self.assertEqual(self.p1._action_len, 8)
        self.assertEqual(self.p1._action_mutate_p, 0.42)
        self.assertEqual(self.p1._condition_cross_p, 0.3)
        self.assertEqual(self.p1._condition_mutate_p, 0.42)
        self.assertIn(Chromosome('2' * 16, '0' * 8, 0.02, True), self.p1.predictors)
        # tests _make_predictors
        self.assertEqual(len(self.p1.predictors), 10)
        self.assertEqual(self.p1._keep, 5)
        self.assertEqual(self.p1.new_genes, self.p1._new_genes_uf)
        self.assertFalse(self.p1.current)

        self.assertEqual(self.p2._num_chroms, 10)
        self.assertEqual(self.p2._condition_len, 16)
        self.assertEqual(self.p2._action_len, 8)
        self.assertEqual(self.p2._action_mutate_p, 0.42)
        self.assertEqual(self.p2._condition_cross_p, 0.3)
        self.assertEqual(self.p2._condition_mutate_p, 0.42)
        self.assertIn(Chromosome('2' * 16, '0' * 8, 0.02, True), self.p2.predictors)
        # tests _make_predictors
        self.assertEqual(len(self.p2.predictors), 10)
        self.assertEqual(self.p2._keep, 5)
        self.assertEqual(self.p2.new_genes, self.p2._new_genes_wf)
        # tests _make_weights()
        self.assertEqual(self.p2._weights[-1], 1)
        self.assertFalse(self.p2.current)

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
        for j in range(10):
            self.p1.predictors[j].used = 1
            self.p1.predictors[j].accuracy = j / 100
        self.p1._match_state(state)
        self.assertEqual(len(self.p1.current), 1)
        self.assertIn(Chromosome('2' * 16, '0' * 8, 0.02, symm=True), self.p1.current)
        self.p1.predictors[0].accuracy = 0.05
        self.p1.predictors[7].accuracy = 0.01
        self.p1._match_state(state)
        self.assertEqual(len(self.p1.current), 2)
        self.assertIn(Chromosome('2221222222222222', '10100011', 0.02, symm=True), self.p1.current)
        self.assertIn(Chromosome('2122212222221221', '01000010', 0.02, symm=True), self.p1.current)

    def test_get_forecast(self):
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
        test 1: Set all Chromosome accuracy to j/100, then match_state chooses Chromosome 0,
        and forecast is Chromosome 0 strategy: 0
        test 2: Set Chromosome 0 accuracy to 0.05, Chromosome 7 accuracy to 0.01, then
        match state chooses Chromosomes 1 and 7, and the forecast is the average strategy:
        (35 + -66)/2 = -15.5
        '''
        state = '1111111111111111'
        for j in range(10):
            self.p1.predictors[j].used = 1
            self.p1.predictors[j].accuracy = j / 100
        self.assertEqual(self.p1.get_forecast(state), 0)
        self.p1.predictors[0].accuracy = 0.05
        self.p1.predictors[7].accuracy = 0.01
        self.assertEqual(self.p1.get_forecast(state), -15.5)

    def test_update_accuracies(self):
        '''
        update_accuracy for a Chromosome is tested elsewhere, here test for a change
        in accuracies by choosing actual != strategy for Chromosomes 1 and 7
        '''
        self.p1.predictors[0].accuracy = 0.05
        self.p1.predictors[7].accuracy = 0.01
        self.p1.current = [self.p1.predictors[0], self.p1.predictors[7]]
        before = (self.p1.predictors[0].accuracy, self.p1.predictors[7].accuracy)
        actual = -35
        self.p1.update_accuracies(actual)
        self.assertNotEqual(before, (self.p1.predictors[0].accuracy, self.p1.predictors[7].accuracy))

    def test_find_winners_uf1(self):
        '''
        Test 1: len(used) < keep
        '''
        for j in range(8):
            if j % 2:
                self.p1.predictors[j].used = 1
                self.p1.predictors[j].accuracy = j / 100
        self.p1._find_winners_uf()
        self.assertEqual(len(self.p1.predictors), 5)
        for i, c in enumerate(self.p1.predictors):
            with self.subTest(c=c):
                if i < 4:
                    self.assertGreater(c.accuracy, 0)
                else:
                    self.assertEqual(c.accuracy, 0)

    def test_find_winners_uf2(self):
        '''
        Test 2: len(used) == keep
        '''
        for j in range(10):
            if j % 2:
                self.p1.predictors[j].used = 1
                self.p1.predictors[j].accuracy = j / 100
        self.p1._find_winners_uf()
        self.assertEqual(len(self.p1.predictors), 5)
        for c in self.p1.predictors:
            with self.subTest(c=c):
                self.assertGreater(c.accuracy, 0)
        self.assertListEqual([c.accuracy for c in self.p1.predictors], [0.01, 0.03, 0.05, 0.07, 0.09])

    def test_find_winners_uf3(self):
        '''
        Test 3: len(used) > keep
        '''
        for j in range(10):
            if j % 2:
                self.p1.predictors[j].used = 1
                self.p1.predictors[j].accuracy = j / 100
        self.p1.predictors[0].used = 1
        self.p1.predictors[0].accuracy = 0.11
        self.p1._find_winners_uf()
        self.assertEqual(len(self.p1.predictors), 5)
        for c in self.p1.predictors:
            with self.subTest(c=c):
                self.assertGreater(c.accuracy, 0)
                self.assertLess(c.accuracy, 0.1)
        self.assertListEqual([c.accuracy for c in self.p1.predictors], [0.01, 0.03, 0.05, 0.07, 0.09])

    def test_find_winners_wf1(self):
        '''
        Test 1: len(used) < keep -> sort on used (descending), then accuracy (ascending)
        '''
        for j in range(8):
            if j % 2:
                self.p1.predictors[j].used = 1
                self.p1.predictors[j].accuracy = j / 100
        self.p1._find_winners_wf()
        self.assertEqual(len(self.p1.predictors), 5)
        for i, c in enumerate(self.p1.predictors):
            with self.subTest(c=c):
                if i < 4:
                    self.assertGreater(c.accuracy, 0)
                else:
                    self.assertEqual(c.accuracy, 0)
        self.assertListEqual([c.accuracy for c in self.p1.predictors], [0.01, 0.03, 0.05, 0.07, 0])

    def test_find_winners_wf2(self):
        '''
        Test 2: len(used) == keep
        '''
        for j in range(10):
            if j % 2:
                self.p1.predictors[j].used = 1
                self.p1.predictors[j].accuracy = j / 100
        self.p1._find_winners_wf()
        self.assertEqual(len(self.p1.predictors), 5)
        for c in self.p1.predictors:
            with self.subTest(c=c):
                self.assertGreater(c.accuracy, 0)
        self.assertListEqual([c.accuracy for c in self.p1.predictors], [0.01, 0.03, 0.05, 0.07, 0.09])

    def test_find_winners_wf3(self):
        '''
        Test 3: len(used) > keep
        '''
        for j in range(10):
            if j % 2:
                self.p1.predictors[j].used = 1
                self.p1.predictors[j].accuracy = j / 100
        self.p1.predictors[0].used = 1
        self.p1.predictors[0].accuracy = 0.11
        self.p1._find_winners_wf()
        self.assertEqual(len(self.p1.predictors), 5)
        for c in self.p1.predictors:
            with self.subTest(c=c):
                self.assertGreater(c.accuracy, 0)
                self.assertLess(c.accuracy, 0.1)
        self.assertListEqual([c.accuracy for c in self.p1.predictors], [0.01, 0.03, 0.05, 0.07, 0.09])
        
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
        c1, c2 = self.p1._mutate(str1, str2, str_len, p_mutate, str_rng)
        self.assertEqual(c1, '1110111111')
        self.assertEqual(c2, '0001000100')
        str3 = '0122120120'
        str4 = '2102102102'
        str_rng = 3
        str_len = len(str3)
        p_mutate = 0.42
        np.random.seed(39)
        random.seed(39)
        c3, c4 = self.p1._mutate(str3, str4, str_len, p_mutate, str_rng)
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
        c1, c2 = self.p1._cross(str1, str2, str_len)
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
        self.p1._check_chrom(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, pred_var)
        # Conditions equal, actions not equal -> use parent_var and append
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p.condition = '2222022220'
        p.action = '00000'
        self.p1._check_chrom(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, parent_var)
        # Conditions equal, actions equal -> no append
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p.condition = '2222022220'
        p.action = '00001'
        self.p1._check_chrom(c, p, pred_var, parent_var)
        self.assertFalse(self.p1.predictors)

    def test_check_chrom2(self):
        pred_var = 1
        parent_var = 2
        # Conditions not equal -> use pred_var and append
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p = Chromosome('2222222222', '00000', 0.02, symm=True)
        c = Chromosome('2222022220', '00001', 0.02, symm=True)
        self.p1._check_chrom2(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, pred_var)
        # Conditions equal, actions not equal -> use parent_var and append
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p.condition = '2222022220'
        p.action = '00000'
        self.p1._check_chrom2(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, parent_var)
        # Conditions equal, actions equal -> use parent_var, append anyway
        self.p1.predictors.clear()
        self.assertFalse(self.p1.predictors)
        p.condition = '2222022220'
        p.action = '00001'
        self.p1._check_chrom2(c, p, pred_var, parent_var)
        self.assertEqual(self.p1.predictors[0].accuracy, parent_var)

    def test_new_genes_uf(self):
        '''
        Trim predictors to 5
        with seeds set to 39, Chromosomes 1 and 2 are selected 
        and 2 suitable children result
        '''
        random.seed(39)
        np.random.seed(39)
        self.p1._num_chroms = 7
        self.p1._action_mutate_p = 0.05
        self.p1._condition_cross_p = 0.3
        self.p1._condition_mutate_p = 0.05
        self.p1.predictors = self.p1.predictors[:5]
        self.assertEqual(len(self.p1.predictors), 5)
        self.p1._new_genes_uf()
        #self.p1.new_genes()
        self.assertEqual(len(self.p1.predictors), 7)
        self.assertEqual(self.p1.predictors[5], Chromosome('2221122121221222', '10100010', 0.02, True))
        self.assertEqual(self.p1.predictors[6], Chromosome('0222222222222222', '11000111', 0.02, True))

    def test_new_genes_wf(self):
        '''
        Trim predictors to 5
        with seeds set to 39, Chromosomes 0 and 1 are selected 
        and 2 suitable children result
        '''
        random.seed(39)
        np.random.seed(39)
        self.p2._num_chroms = 7
        self.p2._action_mutate_p = 0.05
        self.p2._condition_cross_p = 0.3
        self.p2._condition_mutate_p = 0.05
        self.p2._weights = [0.333, 0.6, 0.8, 0.933, 1]
        self.p2.predictors = self.p2.predictors[:5]
        self.assertEqual(len(self.p2.predictors), 5)
        self.p2._new_genes_wf()
        #self.p2.new_genes()
        self.assertEqual(len(self.p2.predictors), 7)
        self.assertEqual(self.p2.predictors[5], Chromosome('1220222222222222', '00010100', 0.02, True))
        self.assertEqual(self.p2.predictors[6], Chromosome('2222221222222222', '01100000', 0.02, True))