import random
import unittest

import numpy as np

from mmabm.shared import Side, OType

from mmabm.trader import MarketMakerL


class TestTrader(unittest.TestCase):
    
    def setUp(self):
        self.l1 = self._makeMML(3001)
        
        self.q1 = {'order_id': 1, 'timestamp': 1, 'type': OType.ADD, 'quantity': 1, 'side': Side.BID,
                   'price': 125}
        
    def _makeMML(self, tid):
        '''
        Two sets of market descriptors: arrival count and order imbalance (net signed order flow)
        arrival count: 16 bits, 8 for previous period and 8 for the previous 5 periods:
            previous period -> one bit each for > 0, 1, 2, 3, 4, 6, 8, 12
            previous 5 periods -> one bit each for >  0, 1, 2, 4, 8, 16, 32, 64
        order imbalance: 24 bits, 12 for previous period and 12 for previous 5 periods:
            previous period -> one bit each for < -8, -4, -3, -2, -1, 0 and > 0, 1, 2, 3, 4, 8
            previous 5 periods -> one bit each for < -16, -8, -6, -4, -2, 0 and > 0, 2, 4, 6, 8, 16
            
        The market maker has a set of predictors (condition/forecast rules) where the condition
        matches the market descriptors (i.e., the market state) and the forecasts are used as inputs
        to the market maker decision making.
        Each market condition is a bit string that coincides with market descriptors with the
        additional possibility of "don't care" (==2). 
        Each market condition has an associated forecast
        arrival count: 5 bits -> 2^5 - 1 = 31 for a range of 0 - 31
        order imbalance: 6 bits -> lhs bit is +1/-1 and 2^5 - 1 = 31 for a range of -31 - +31
        
        Each market maker receives 100 genes for each of the two sets of market descriptors and
        25 genes for the arrival forecast action rule.
        Examples:
        arrival count: 1111100011111100 -> >4 for previous period and >8 for previous 5 periods
        arrival count gene -> 2222102222221122: 01010 
            this gene matches on the "do care" (0 or 1) bits and has "don't care" for the remaining
            bits. It forecasts an arrival count of 10 (0*16 + 1*8 + 0*4 + 1*2 + 0*1).
        order imbalance: 011111000000011111000000 - < -4 for previous period and < -8 for previous
        5 periods
        order imbalance gene: 222221022222222122222012: 010010
            this gene does not match the market state in position 23 and forecasts an order
            imbalance of +18 (+1*(1*16 + 0*8 + 0*4 + 1*2 + 0*1))
            
        The arrival count forecast acts as a condition/action rule where the condition matches the
        arrival count forecast and the action adjusts the bid and ask prices:
        arrival count forecast: 5 bits -> 2^5 - 1 = 31 for a range of 0 - 31
        action: 4 bits  -> lhs bit is +1/-1 and 2^3 - 1 = 7 for a range of -7 - +7
        Example:
        arrival count forecast -> 01010
        arrival count gene -> 02210: 0010
            this gene matches the arrival count forecast and adjusts the bid (or ask) by (+1*(0*4 + 1*2 + 0*1) = +2.
        '''
        random.seed(39)
        np.random.seed(39)
        gene_n1 = 100
        gene_n2 = 25
        arr_cond_n = 16
        oi_cond_n = 24
        bidp_cond_n = 5
        askp_cond_n = 5
        arr_fcst_n = 5
        oi_fcst_n = 6
        bidp_adj_n = 4
        askp_adj_n = 4
        probs = [0.05, 0.05, 0.9]
        
        arr_genes = {}
        oi_genes = {}
        bidp_genes = {}
        askp_genes = {}
        genes = tuple([oi_genes, arr_genes, askp_genes, bidp_genes])
        while len(arr_genes) < gene_n1:
            gk = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), arr_cond_n, p=probs))
            gv = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), arr_fcst_n))
            arr_genes.update({gk: gv})
        while len(oi_genes) < gene_n1:
            gk = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), oi_cond_n, p=probs))
            gv = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), oi_fcst_n))
            oi_genes.update({gk: gv})
        while len(bidp_genes) < gene_n2:
            gk = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), bidp_cond_n, p=probs))
            gv = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), bidp_adj_n))
            bidp_genes.update({gk: gv})
        while len(askp_genes) < gene_n2:
            gk = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), askp_cond_n, p=probs))
            gv = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), askp_adj_n))
            askp_genes.update({gk: gv})
        maxq = 5
        a = b = c = 1
        return MarketMakerL(tid, maxq, a, b, c, genes)
    
    ''' Strategy Construction Tests '''    
    def test_make_oi_strat2(self):
        ''' The OI strat has 100 genes each with 24 bits '''
        self.assertEqual(self.l1._oi_len, 24)
        self.assertEqual(len(self.l1._arr_strat), 100)
        #print(self.l1._oi_strat, self.l1._oi_len, len(self.l1._arr_strat))
        
    def test_make_arr_strat2(self):
        ''' The Arr strat has 100 genes each with 16 bits '''
        self.assertEqual(self.l1._arr_len, 16)
        self.assertEqual(len(self.l1._arr_strat), 100)
        #print(self.l1._arr_strat, self.l1._arr_len, len(self.l1._oi_strat))
        
    def test_make_bidask_strat2(self):
        ''' The Bid and Ask Adj strats have 25 genes each with 5 bits '''
        self.assertEqual(self.l1._ask_len, 5)
        self.assertEqual(len(self.l1._askadj_strat), 25)
        self.assertEqual(self.l1._bid_len, 5)
        self.assertEqual(len(self.l1._bidadj_strat), 25)
        #print(self.l1._askadj_strat, self.l1._ask_len, len(self.l1._askadj_strat))
        #print(self.l1._bidadj_strat, self.l1._bid_len, len(self.l1._bidadj_strat))
        
    @unittest.skip('Takes too long to run every time')
    def test_make_oi_strat(self):
        ''' Test for proper conversion from bitstring to integer '''
        for i in self.l1._oi_strat.keys():
            with self.subTest(i=i):
                self.assertEqual(int(self.l1._oi_strat[i]['action'][1:], 2), abs(self.l1._oi_strat[i]['strategy']))
                if self.l1._oi_strat[i]['strategy'] != 0:
                    self.assertEqual(int(self.l1._oi_strat[i]['action'][0]), self.l1._oi_strat[i]['strategy']>0)
                self.assertEqual(self.l1._oi_strat[i]['accuracy'], [0, 0, 0])
                
    @unittest.skip('Takes too long to run every time')      
    def test_make_arr_strat(self):
        ''' Test for proper conversion from bitstring to integer '''
        for i in self.l1._arr_strat.keys():
            with self.subTest(i=i):
                self.assertEqual(int(self.l1._arr_strat[i]['action'], 2), self.l1._arr_strat[i]['strategy'])
                self.assertEqual(self.l1._arr_strat[i]['accuracy'], [0, 0, 0])
                
    @unittest.skip('Takes too long to run every time')        
    def test_make_bidask_strat(self):
        ''' Test for proper conversion from bitstring to integer '''
        #ask strategy
        for i in self.l1._askadj_strat.keys():
            with self.subTest(i=i):
                self.assertEqual(int(self.l1._askadj_strat[i]['action'][1:], 2), abs(self.l1._askadj_strat[i]['strategy']))
                if self.l1._askadj_strat[i]['strategy'] != 0:
                    self.assertEqual(int(self.l1._askadj_strat[i]['action'][0]), self.l1._askadj_strat[i]['strategy']>0)
                self.assertEqual(self.l1._askadj_strat[i]['profitability'], [0, 0, 0])
        #bid strategy        
        for i in self.l1._bidadj_strat.keys():
            with self.subTest(i=i):
                self.assertEqual(int(self.l1._bidadj_strat[i]['action'][1:], 2), abs(self.l1._bidadj_strat[i]['strategy']))
                if self.l1._bidadj_strat[i]['strategy'] != 0:
                    self.assertEqual(int(self.l1._bidadj_strat[i]['action'][0]), self.l1._bidadj_strat[i]['strategy']>0)
                self.assertEqual(self.l1._bidadj_strat[i]['profitability'], [0, 0, 0])
        
    ''' Strategy Matching Tests '''
    def test_match_oi_strat2(self):
        ''' With seeds == 39, '221212222222222222020222' is the sole winning strategy with a max strength == 4  '''
        #oi_state is 24 bits
        signal = '011111000000011111000000'
        self.l1._match_oi_strat2(signal)
        self.assertEqual(self.l1._current_oi_strat[0], '221212222222222222020222')
        self.assertTrue(all([(self.l1._current_oi_strat[0][x] == signal[x] or self.l1._current_oi_strat[0][x] == '2') for x in range(self.l1._oi_len)]))
        self.assertEqual(sum([self.l1._current_oi_strat[0][x] == signal[x] for x in range(self.l1._oi_len)]), 4)
        
    def test_match_arr_strat2(self):
        pass
    
    def test_match_ask_strat(self):
        pass
    
    def test_match_bid_strat(self):
        pass
    
    ''' Accuracy/Profitability Update Tests '''
    def test_update_oi_acc(self):
        pass
    
    def test_update_arr_acc(self):
        pass
    
    def test_update_profits(self):
        pass
    
    ''' Trade Handling Tests '''
    def test_confirm_trade_local(self):
        pass 
     
    ''' Order Construction Tests '''    
    def test_make_add_quote(self):
        ''' Takes 4 inputs, increments the quote sequence and generates a dict '''
        time = 1
        side = Side.ASK
        price = 125
        quantity = 5
        self.assertFalse(self.l1._quote_sequence)
        expected = {'order_id': 1, 'trader_id': self.l1.trader_id, 'timestamp': 1, 'type': OType.ADD, 
                    'quantity': quantity, 'side': Side.ASK, 'price': 125}
        self.assertDictEqual(self.l1._make_add_quote(time, side, price, quantity), expected)
        
    def test_make_cancel_quote(self):
        ''' Takes a quote and current timestamp, resets the timestamp to current,
        and updates type to CANCEL '''
        self.q1['trader_id'] = self.l1.trader_id
        expected = {'order_id': 1, 'trader_id': self.l1.trader_id, 'timestamp': 2, 'type': OType.CANCEL, 
                    'quantity': 1, 'side': Side.BID, 'price': 125}
        self.assertDictEqual(self.l1._make_cancel_quote(self.q1, 2), expected)
        
    ''' Orderbook Bookkeeping Tests'''
    def test_add_order(self):
        pass
    
    def test_remove_order(self):
        pass
    
    def test_modify_order(self):
        pass
    
    ''' Orderbook Update Tests '''    
    def test_update_midpoint(self):
        pass
    
    def test_make_spread(self):
        pass
    
    def test_update_ask_book(self):
        pass
    
    def test_update_bid_book(self):
        pass
    
    def test_process_signal(self):
        pass
        