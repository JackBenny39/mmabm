import random
import unittest

import numpy as np

from mmabm.shared import Side, OType

from mmabm.learner import MarketMakerL


class TestTrader(unittest.TestCase):
    
    def setUp(self):
        self.l1 = self._makeMML(3001, 1)
        
        self.q1 = {'order_id': 1, 'timestamp': 1, 'type': OType.ADD, 'quantity': 1, 'side': Side.BID,
                   'price': 125}
        
        self.q1_buy = {'order_id': 1,'timestamp': 2, 'type': OType.ADD, 'quantity': 1, 'side': Side.BID,
                       'price': 50}
        self.q2_buy = {'order_id': 2, 'timestamp': 3, 'type': OType.ADD, 'quantity': 1, 'side': Side.BID,
                       'price': 50}
        self.q3_buy = {'order_id': 1, 'timestamp': 4, 'type': OType.ADD, 'quantity': 3, 'side': Side.BID,
                       'price': 49}
        self.q4_buy = {'order_id': 1, 'timestamp': 5, 'type': OType.ADD, 'quantity': 3, 'side': Side.BID,
                       'price': 47}
        self.q1_sell = {'order_id': 3, 'timestamp': 2, 'type': OType.ADD, 'quantity': 1, 'side': Side.ASK,
                        'price': 52}
        self.q2_sell = {'order_id': 4, 'timestamp': 3, 'type': OType.ADD, 'quantity': 1, 'side': Side.ASK,
                        'price': 52}
        self.q3_sell = {'order_id': 2, 'timestamp': 4, 'type': OType.ADD, 'quantity': 3, 'side': Side.ASK,
                        'price': 53}
        self.q4_sell = {'order_id': 2, 'timestamp': 5, 'type': OType.ADD, 'quantity': 3, 'side': Side.ASK,
                        'price': 55}
        
    def _makeMML(self, tid, arrInt):
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
        spr_cond_n = 5
        arr_fcst_n = 5
        oi_fcst_n = 6
        spr_adj_n = 4
        probs = [0.05, 0.05, 0.9]
        
        arr_genes = {}
        oi_genes = {}
        spread_genes = {}
        genes = tuple([oi_genes, arr_genes, spread_genes])
        while len(arr_genes) < gene_n1:
            gk = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), arr_cond_n, p=probs))
            gv = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), arr_fcst_n))
            arr_genes.update({gk: gv})
        while len(oi_genes) < gene_n1:
            gk = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), oi_cond_n, p=probs))
            gv = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), oi_fcst_n))
            oi_genes.update({gk: gv})
        while len(spread_genes) < gene_n2:
            gk = ''.join(str(x) for x in np.random.choice(np.arange(0, 3), spr_cond_n, p=probs))
            gv = ''.join(str(x) for x in np.random.choice(np.arange(0, 2), spr_adj_n))
            spread_genes.update({gk: gv})
        maxq = 5
        a = b = 1
        c = -1
        keeper = 0.8
        mutate_pct = 0.03
        return MarketMakerL(tid, maxq, arrInt, a, b, c, genes, keeper, mutate_pct)
    
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
        
    def test_make_spread_strat2(self):
        ''' The Spread Adj strat has 25 genes each with 5 bits '''
        self.assertEqual(self.l1._spr_len, 5)
        self.assertEqual(len(self.l1._spradj_strat), 25)
        #print(self.l1._spradj_strat, self.l1._spr_len, len(self.l1._spradj_strat))
        
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
    def test_make_spread_strat(self):
        ''' Test for proper conversion from bitstring to integer '''
        # spread strategy
        for i in self.l1._spradj_strat.keys():
            with self.subTest(i=i):
                self.assertEqual(int(self.l1._spradj_strat[i]['action'][1:], 2), abs(self.l1._spradj_strat[i]['strategy']))
                if self.l1._spradj_strat[i]['strategy'] != 0:
                    self.assertEqual(int(self.l1._spradj_strat[i]['action'][0]), self.l1._spradj_strat[i]['strategy']>0)
                self.assertEqual(self.l1._spradj_strat[i]['rr_spread'], [0, 0, 0])
        
    ''' Strategy Matching Tests '''
    def test_match_oi_strat2(self):
        ''' With seeds == 39, '221212222222222222020222' is the sole winning strategy with a max strength == 4  '''
        #oi_state is 24 bits
        signal = '011111000000011111000000'
        self.l1._match_oi_strat2(signal)
        self.assertEqual(self.l1._current_oi_strat[0], '221212222222222222020222')
        self.assertTrue(all([(self.l1._current_oi_strat[0][x] == signal[x] or self.l1._current_oi_strat[0][x] == '2') for x in range(self.l1._oi_len)]))
        self.assertEqual(sum([self.l1._current_oi_strat[0][x] == signal[x] for x in range(self.l1._oi_len)]), 4)
        # Another winner with strength == 4 could be '212212222222222222020222'
        self.l1._oi_strat['212212222222222222020222'] = {'action': 'xxxxx', 'strategy': 999, 'accuracy': [0, 0, 1]}
        # Set '221212222222222222020222' accuracy to greater than new strat accuracy
        self.l1._oi_strat['221212222222222222020222']['accuracy'][-1] = 2
        self.l1._match_oi_strat2(signal)
        self.assertEqual(self.l1._current_oi_strat[0], '212212222222222222020222')
        self.assertTrue(all([(self.l1._current_oi_strat[0][x] == signal[x] or self.l1._current_oi_strat[0][x] == '2') for x in range(self.l1._oi_len)]))
        self.assertEqual(sum([self.l1._current_oi_strat[0][x] == signal[x] for x in range(self.l1._oi_len)]), 4)
        # If they had the same strength and accuracy, both would be returned
        self.l1._oi_strat['221212222222222222020222']['accuracy'][-1] = 1
        self.l1._match_oi_strat2(signal)
        for j in ['212212222222222222020222', '221212222222222222020222']:
            self.assertTrue(j in self.l1._current_oi_strat)
        
    def test_match_arr_strat2(self):
        ''' With seeds == 39, '1222102221222222' is the winning strategy with a max strength == 4  '''
        signal = '1111100011111100'
        self.l1._match_arr_strat2(signal)
        self.assertEqual(self.l1._current_arr_strat, '1222102221222222')
        self.assertTrue(all([(self.l1._current_arr_strat[x] == signal[x] or self.l1._current_arr_strat[x] == '2') for x in range(self.l1._arr_len)]))
        self.assertEqual(sum([self.l1._current_arr_strat[x] == signal[x] for x in range(self.l1._arr_len)]), 4)
        
        # Another winner with strength == 4 could be '2122102221222222'
        self.l1._arr_strat['2122102221222222'] = {'action': 'xxxxx', 'strategy': 999, 'accuracy': [0, 0, 1]}
        # Set '1222102221222222' accuracy to greater than new strat accuracy
        self.l1._arr_strat['1222102221222222']['accuracy'][-1] = 2
        self.l1._match_arr_strat2(signal)
        self.assertEqual(self.l1._current_arr_strat, '2122102221222222')
        self.assertTrue(all([(self.l1._current_arr_strat[x] == signal[x] or self.l1._current_arr_strat[x] == '2') for x in range(self.l1._arr_len)]))
        self.assertEqual(sum([self.l1._current_arr_strat[x] == signal[x] for x in range(self.l1._arr_len)]), 4)
        # If they had the same strength and accuracy, only one would be returned
        self.l1._arr_strat['1222102221222222']['accuracy'][-1] = 1
        self.l1._match_arr_strat2(signal)
        self.assertFalse('2122102221222222' in self.l1._current_arr_strat)
        self.assertTrue('1222102221222222' in self.l1._current_arr_strat)
    
    def test_match_spread_strat(self):
        ''' With seeds == 39, ['21220', '22020', '02022', '21212', '22210', '02212'] are the winning strategies with a max strength == 2  '''
        signal = '01010'
        self.l1._match_spread_strat(signal)
        for j in ['21220', '22020', '02022', '21212', '22210', '02212']:
            self.assertTrue(j in self.l1._current_spradj_strat)
        for i in self.l1._current_spradj_strat:
            with self.subTest(i=i):
                self.assertTrue(all([(i[x] == signal[x] or i[x] == '2') for x in range(self.l1._spr_len)]))
                self.assertEqual(sum([i[x] == signal[x] for x in range(self.l1._spr_len)]), 2)
        # Winner '01222' - set rr_spread higher
        self.l1._spradj_strat['01222'] = {'action': 'xxxxx', 'strategy': 999, 'rr_spread': [0, 0, 1]}
        self.l1._match_spread_strat(signal)
        self.assertEqual(len(self.l1._current_spradj_strat), 1)
        self.assertEqual(self.l1._current_spradj_strat[0], '01222')
        self.assertTrue(all([(self.l1._current_spradj_strat[0][x] == signal[x] or self.l1._current_spradj_strat[0][x] == '2') for x in range(self.l1._spr_len)]))
        self.assertEqual(sum([self.l1._current_spradj_strat[0][x] == signal[x] for x in range(self.l1._spr_len)]), 2)
    
    ''' Accuracy/Profitability Update Tests '''
    def test_update_oi_acc(self):
        self.l1._oi_strat['221212222222222222020222']['accuracy'][0] = 10
        self.l1._oi_strat['221212222222222222020222']['accuracy'][1] = 10
        self.l1._oi_strat['221212222222222222020222']['accuracy'][-1] = 1
        self.l1._oi_strat['221212222222222222020222']['strategy'] = 4
        self.l1._current_oi_strat = ['221212222222222222020222']
        actual = 6
        self.l1._update_oi_acc(actual)
        self.assertListEqual(self.l1._oi_strat['221212222222222222020222']['accuracy'], [12, 11, 12/11])
    
    def test_update_arr_acc(self):
        self.l1._arr_strat['1222102221222222']['accuracy'][0] = 10
        self.l1._arr_strat['1222102221222222']['accuracy'][1] = 10
        self.l1._arr_strat['1222102221222222']['accuracy'][-1] = 1
        self.l1._arr_strat['1222102221222222']['strategy'] = 4
        self.l1._current_arr_strat = '1222102221222222'
        actual = 6
        self.l1._update_arr_acc(actual)
        self.assertListEqual(self.l1._arr_strat['1222102221222222']['accuracy'], [12, 11, 12/11])
    
    def test_update_rspr(self):
        self.l1._spradj_strat['21220']['rr_spread'][0] = 10000
        self.l1._spradj_strat['21220']['rr_spread'][1] = 1000
        self.l1._spradj_strat['21220']['rr_spread'][-1] = 10
        self.l1._current_spradj_strat = ['21220']
        mid = 1000
        self.l1._last_buy_prices = [998, 999]
        self.l1._last_sell_prices = [1001, 1002]
        self.l1._update_rspr(mid)
        self.assertListEqual(self.l1._spradj_strat['21220']['rr_spread'], [10006, 1004, 10006/1004])
    
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
        '''
        add_order_to_book() impacts _bid_book and _bid_book_prices or _ask_book and _ask_book_prices
        Add two buy orders, then two sell orders
        '''
        # 2 buy orders
        self.assertFalse(self.l1._bid_book_prices)
        self.assertFalse(self.l1._bid_book)
        self.l1._add_order(self.q1_buy)
        self.assertTrue(50 in self.l1._bid_book_prices)
        self.assertTrue(50 in self.l1._bid_book.keys())
        self.assertEqual(self.l1._bid_book[50]['num_orders'], 1)
        self.assertEqual(self.l1._bid_book[50]['size'], 1)
        self.assertEqual(self.l1._bid_book[50]['order_ids'][0], 1)
        del self.q1_buy['type']
        self.assertDictEqual(self.l1._bid_book[50]['orders'][1], self.q1_buy)
        self.l1._add_order(self.q2_buy)
        self.assertEqual(self.l1._bid_book[50]['num_orders'], 2)
        self.assertEqual(self.l1._bid_book[50]['size'], 2)
        self.assertEqual(self.l1._bid_book[50]['order_ids'][1], 2)
        del self.q2_buy['type']
        self.assertDictEqual(self.l1._bid_book[50]['orders'][2], self.q2_buy)
        # 2 sell orders
        self.assertFalse(self.l1._ask_book_prices)
        self.assertFalse(self.l1._ask_book)
        self.l1._add_order(self.q1_sell)
        self.assertTrue(52 in self.l1._ask_book_prices)
        self.assertTrue(52 in self.l1._ask_book.keys())
        self.assertEqual(self.l1._ask_book[52]['num_orders'], 1)
        self.assertEqual(self.l1._ask_book[52]['size'], 1)
        self.assertEqual(self.l1._ask_book[52]['order_ids'][0], 3)
        del self.q1_sell['type']
        self.assertDictEqual(self.l1._ask_book[52]['orders'][3], self.q1_sell)
        self.l1._add_order(self.q2_sell)
        self.assertEqual(self.l1._ask_book[52]['num_orders'], 2)
        self.assertEqual(self.l1._ask_book[52]['size'], 2)
        self.assertEqual(self.l1._ask_book[52]['order_ids'][1], 4)
        del self.q2_sell['type']
        self.assertDictEqual(self.l1._ask_book[52]['orders'][4], self.q2_sell)
    
    def test_remove_order(self):
        '''
        _remove_order() impacts _bid_book and _bid_book_prices or _ask_book and _ask_book_prices
        Add two  orders, remove the second order twice
        '''
        # buy orders
        self.l1._add_order(self.q1_buy)
        self.l1._add_order(self.q2_buy)
        self.assertTrue(50 in self.l1._bid_book_prices)
        self.assertTrue(50 in self.l1._bid_book.keys())
        self.assertEqual(self.l1._bid_book[50]['num_orders'], 2)
        self.assertEqual(self.l1._bid_book[50]['size'], 2)
        self.assertEqual(len(self.l1._bid_book[50]['order_ids']), 2)
        # remove first order
        self.l1._remove_order(Side.BID, 50, 1)
        self.assertEqual(self.l1._bid_book[50]['num_orders'], 1)
        self.assertEqual(self.l1._bid_book[50]['size'], 1)
        self.assertEqual(len(self.l1._bid_book[50]['order_ids']), 1)
        self.assertFalse(1 in self.l1._bid_book[50]['orders'].keys())
        self.assertTrue(50 in self.l1._bid_book_prices)
        # remove second order
        self.l1._remove_order(Side.BID, 50, 2)
        self.assertFalse(self.l1._bid_book_prices)
        self.assertEqual(self.l1._bid_book[50]['num_orders'], 0)
        self.assertEqual(self.l1._bid_book[50]['size'], 0)
        self.assertEqual(len(self.l1._bid_book[50]['order_ids']), 0)
        self.assertFalse(2 in self.l1._bid_book[50]['orders'].keys())
        self.assertFalse(50 in self.l1._bid_book_prices)
        # remove second order again
        self.l1._remove_order(Side.BID, 50, 2)
        self.assertFalse(self.l1._bid_book_prices)
        self.assertEqual(self.l1._bid_book[50]['num_orders'], 0)
        self.assertEqual(self.l1._bid_book[50]['size'], 0)
        self.assertEqual(len(self.l1._bid_book[50]['order_ids']), 0)
        self.assertFalse(2 in self.l1._bid_book[50]['orders'].keys())
        # sell orders
        self.l1._add_order(self.q1_sell)
        self.l1._add_order(self.q2_sell)
        self.assertTrue(52 in self.l1._ask_book_prices)
        self.assertTrue(52 in self.l1._ask_book.keys())
        self.assertEqual(self.l1._ask_book[52]['num_orders'], 2)
        self.assertEqual(self.l1._ask_book[52]['size'], 2)
        self.assertEqual(len(self.l1._ask_book[52]['order_ids']), 2)
        # remove first order
        self.l1._remove_order(Side.ASK, 52, 3)
        self.assertEqual(self.l1._ask_book[52]['num_orders'], 1)
        self.assertEqual(self.l1._ask_book[52]['size'], 1)
        self.assertEqual(len(self.l1._ask_book[52]['order_ids']), 1)
        self.assertFalse(3 in self.l1._ask_book[52]['orders'].keys())
        self.assertTrue(52 in self.l1._ask_book_prices)
        # remove second order
        self.l1._remove_order(Side.ASK, 52, 4)
        self.assertFalse(self.l1._ask_book_prices)
        self.assertEqual(self.l1._ask_book[52]['num_orders'], 0)
        self.assertEqual(self.l1._ask_book[52]['size'], 0)
        self.assertEqual(len(self.l1._ask_book[52]['order_ids']), 0)
        self.assertFalse(4 in self.l1._ask_book[52]['orders'].keys())
        self.assertFalse(52 in self.l1._ask_book_prices)
        # remove second order again
        self.l1._remove_order(Side.ASK, 52, 4)
        self.assertFalse(self.l1._ask_book_prices)
        self.assertEqual(self.l1._ask_book[52]['num_orders'], 0)
        self.assertEqual(self.l1._ask_book[52]['size'], 0)
        self.assertEqual(len(self.l1._ask_book[52]['order_ids']), 0)
        self.assertFalse(4 in self.l1._ask_book[52]['orders'].keys())
    
    def test_modify_order(self):
        '''
        _modify_order() primarily impacts _bid_book or _ask_book 
        _modify_order() could impact _bid_book_prices or _ask_book_prices if the order results 
        in removing the full quantity with a call to _remove_order() 
        Add 1 order, remove partial, then remainder
        '''
        # Buy order
        q1 = {'order_id': 1, 'timestamp': 5, 'type': OType.ADD, 'quantity': 2, 'side': Side.BID, 'price': 50}
        self.l1._add_order(q1)
        self.assertEqual(self.l1._bid_book[50]['size'], 2)
        # remove 1
        self.l1._modify_order(Side.BID, 1, 1, 50)
        self.assertEqual(self.l1._bid_book[50]['size'], 1)
        self.assertEqual(self.l1._bid_book[50]['orders'][1]['quantity'], 1)
        self.assertTrue(self.l1._bid_book_prices)
        # remove remainder
        self.l1._modify_order(Side.BID, 1, 1, 50)
        self.assertFalse(self.l1._bid_book_prices)
        self.assertEqual(self.l1._bid_book[50]['num_orders'], 0)
        self.assertEqual(self.l1._bid_book[50]['size'], 0)
        self.assertFalse(1 in self.l1._bid_book[50]['orders'].keys())
        # Sell order
        q2 = {'order_id': 2, 'timestamp': 5, 'type': OType.ADD, 'quantity': 2, 'side': Side.ASK, 'price': 50}
        self.l1._add_order(q2)
        self.assertEqual(self.l1._ask_book[50]['size'], 2)
        # remove 1
        self.l1._modify_order(Side.ASK, 1, 2, 50)
        self.assertEqual(self.l1._ask_book[50]['size'], 1)
        self.assertEqual(self.l1._ask_book[50]['orders'][2]['quantity'], 1)
        self.assertTrue(self.l1._ask_book_prices)
        # remove remainder
        self.l1._modify_order(Side.ASK, 1, 2, 50)
        self.assertFalse(self.l1._ask_book_prices)
        self.assertEqual(self.l1._ask_book[50]['num_orders'], 0)
        self.assertEqual(self.l1._ask_book[50]['size'], 0)
        self.assertFalse(2 in self.l1._ask_book[50]['orders'].keys())
    
    ''' Trade Handling Tests '''
    def test_confirm_trade_local(self):
        # _cash_flow and _delta_inv start at 0, _last_buy_prices and last_sell_prices are empty
        self.assertFalse(self.l1._last_buy_prices)
        self.assertFalse(self.l1._last_sell_prices)
        self.assertEqual(self.l1._cash_flow, 0)
        self.assertEqual(self.l1._delta_inv, 0)
        # add some orders
        q1 = {'order_id': 1, 'timestamp': 5, 'type': OType.ADD, 'quantity': 5, 'side': Side.BID, 'price': 995}
        q2 = {'order_id': 2, 'timestamp': 5, 'type': OType.ADD, 'quantity': 5, 'side': Side.ASK, 'price': 1005}
        self.l1._add_order(q1)
        self.l1._add_order(q2)
        # Market maker buys
        confirm1 = {'timestamp': 20, 'trader': 3001, 'order_id': 1, 'quantity': 1, 'side': Side.BID, 'price': 995}
        self.l1.confirm_trade_local(confirm1)
        self.assertListEqual(self.l1._last_buy_prices, [995])
        self.assertEqual(self.l1._cash_flow, -995)
        self.assertEqual(self.l1._delta_inv, 1)
        self.assertEqual(self.l1._bid_book[995]['num_orders'], 1)
        self.assertEqual(self.l1._bid_book[995]['size'], 4)
        confirm2 = {'timestamp': 22, 'trader': 3001, 'order_id': 1, 'quantity': 4, 'side': Side.BID, 'price': 995}
        self.l1.confirm_trade_local(confirm2)
        self.assertListEqual(self.l1._last_buy_prices, [995, 995])
        self.assertEqual(self.l1._cash_flow, -4975)
        self.assertEqual(self.l1._delta_inv, 5)
        self.assertFalse(self.l1._bid_book_prices)
        # Market maker sells
        confirm3 = {'timestamp': 20, 'trader': 3001, 'order_id': 2, 'quantity': 1, 'side': Side.ASK, 'price': 1005}
        self.l1.confirm_trade_local(confirm3)
        self.assertListEqual(self.l1._last_sell_prices, [1005])
        self.assertEqual(self.l1._cash_flow, -3970)
        self.assertEqual(self.l1._delta_inv, 4)
        self.assertEqual(self.l1._ask_book[1005]['num_orders'], 1)
        self.assertEqual(self.l1._ask_book[1005]['size'], 4)
        confirm4 = {'timestamp': 22, 'trader': 3001, 'order_id': 2, 'quantity': 4, 'side': Side.ASK, 'price': 1005}
        self.l1.confirm_trade_local(confirm4)
        self.assertListEqual(self.l1._last_sell_prices, [1005, 1005])
        self.assertEqual(self.l1._cash_flow, 50)
        self.assertEqual(self.l1._delta_inv, 0)
        self.assertFalse(self.l1._ask_book_prices)

    ''' Orderbook Update Tests '''    
    def test_update_midpoint(self):
        ''' With seeds == 39, '221212222222222222020222' is the sole winning strategy with a max strength == 4 -> action == -3 '''
        self.l1._mid = 1000
        self.l1._delta_inv = 3
        #oi_state is 24 bits
        signal = '011111000000011111000000'
        self.l1._update_midpoint(signal)
        # new mid = old mid - 3 + (-1*3)
        self.assertEqual(self.l1._mid, 994)
    
    def test_make_spread(self):
        ''' With seeds == 39, '1222102221222222' is the winning arr strategy with a max strength == 4 -> action == '01000' (8)
        _match_spread_strat('01000') returns ['21220', '22020', '02022'] with an actions of ['0000', '1100', '1111'] -> 
        strategy == [0, 4, 7] for an average of 3.67
        '''
        self.assertFalse(self.l1._ask)
        self.assertFalse(self.l1._bid)
        self.l1._mid = 1000
        arr_signal = '1222102221222222'
        vol = 4
        self.l1._make_spread(arr_signal, vol)
        # spradj = 3.67
        # ask = 1000 + round(max(1*4, 1) + 3.67/2) = 1006
        # bid = 1000 - round(max(1*4, 1) + 3.67/2) = 994
        self.assertEqual(self.l1._bid, 994)
        self.assertEqual(self.l1._ask, 1006)
        
    def test_process_cancels(self):
        ''' If desired ask > current best ask, cancel current ask orders with prices < new best ask '''
        # Create asks from 1005 - 1035
        for p in range(1005, 1036):
            self.l1._add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(1005, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        # Create bids from 960 - 990
        for p in range(960, 991):
            self.l1._add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 991):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        # case 1a: new ask = 1000, new bid = 995 -> no new cancels
        self.l1._ask = 1000
        self.l1._bid = 995
        self.l1._process_cancels(6)
        self.assertFalse(self.l1.cancel_collector)
        # case 2a: new ask = 1008 -> cancel 3 prices: 1005, 1006, 1007
        self.l1._ask = 1008
        self.l1._bid = 995
        self.l1._process_cancels(7)
        for p in range(1008, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        for p in range(1005, 1008):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._ask_book_prices)
        self.assertEqual(len(self.l1.cancel_collector), 3)
        # case 2b: new bid = 987 -> cancel 988, 989, 990
        self.l1._ask = 1000
        self.l1._bid = 987
        self.l1._process_cancels(8)
        for p in range(960, 988):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        for p in range(988, 991):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._bid_book_prices)
        self.assertEqual(len(self.l1.cancel_collector), 3)

    ''' Several cases:
    1. _ask > prevailing worst ask (ask book empty due to canceling first)
    2. _ask > prevailing best bid: add new ask orders from _ask and up
    3. _ask <= prevailing best bid: add new ask orders from prevailing best bid+1 and up
    4. _ask == current best ask: check for max size and add size if necessary
    Also, price range should always be between best ask + 20 and best ask + 60
    '''
    def test_update_ask_book1(self):
        ''' 1. _ask > prevailing worst ask (ask book empty due to canceling first) '''
        self.assertFalse(self.l1._ask_book_prices)
        self.l1._ask = 1000
        tob_bid = 995
        self.l1._update_ask_book(6, tob_bid)
        # case 1: Add orders from 1000 -> 1039
        for p in range(1000, 1040):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 40)
    
    def test_update_ask_book2(self):
        ''' 2. _ask > prevailing best bid: add new ask orders from _ask and up '''
        # Create asks from 1005 - 1035
        for p in range(1005, 1036):
            self.l1._add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(1005, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        # case 2: _ask = 1000, best_bid = 995 -> add 5 new prices: 1000 - 1004
        self.l1._ask = 1000
        tob_bid = 995
        self.l1._update_ask_book(6, tob_bid)
        for p in range(1000, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 5)
        
    def test_update_ask_book3(self):
        ''' 3. _ask <= prevailing best bid: add new ask orders from prevailing best bid+1 and up '''
        for p in range(1000, 1036):
            self.l1._add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(1000, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        # case 3: _ask = 990 but best_bid = 995 -> add 4 new prices: 996 - 999
        self.l1._ask = 990
        tob_bid = 995
        self.l1._update_ask_book(7, tob_bid)
        for p in range(996, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        for p in range(990, 996):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._ask_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 4)
        
    def test_update_ask_book4(self):
        ''' 4. _ask == current best ask: check for max size and add size if necessary '''
        for p in range(996, 1036):
            self.l1._add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(996, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        # case 4: new ask size == 2 -> replenish size to 5 with new add order
        self.l1._ask = 996
        tob_bid = 990
        self.l1._modify_order(Side.ASK, 3, 1, 996)
        self.assertEqual(self.l1._ask_book[996]['orders'][1]['quantity'], 2)
        self.assertEqual(self.l1._ask_book[996]['size'], 2)
        self.assertEqual(self.l1._ask_book[996]['num_orders'], 1)
        self.l1.quote_collector.clear() # happens in process_order
        self.l1._update_ask_book(8, tob_bid)
        self.assertEqual(self.l1._ask_book[996]['size'], 5)
        self.assertEqual(self.l1._ask_book[996]['num_orders'], 2)
        self.assertEqual(len(self.l1.quote_collector), 1)
        
    def test_update_ask_book5(self):
        ''' Also, price range should always be between best ask + 20 and best ask + 60 '''
        # make best ask == 1020 -> add orders to the other end of the book to make 40 prices
        for p in range(1020, 1036):
            self.l1._add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(1020, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        tob_bid = 990
        self.l1._ask = 1020
        self.l1._update_ask_book(10, tob_bid)
        for p in range(1020, 1059):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        # make best ask == 980 -> cancel orders on the other end of the book to make 40 prices
        self.l1._bid = 975
        self.l1._ask = 980
        tob_bid = 975
        self.l1._update_ask_book(10, tob_bid)
        for p in range(980, 1019):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        for p in range(1020, 1059):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._ask_book_prices)
                
    ''' Several cases:
    1. _bid < prevailing worst bid (bid book empty due to canceling first)
    2. _bid < prevailing best ask: add new bid orders from _bid and down
    3. _bid >= prevailing best ask: add new bid orders from prevailing best ask-1 and down
    4. _bid == current best bid: check for max size and add size if necessary
    Also, price range should always be between best bid - 20 and best bid - 60
    '''
    def test_update_bid_book1(self):
        ''' 1. _bid < prevailing worst bid (bid book empty due to canceling first) '''
        self.assertFalse(self.l1._bid_book_prices)
        self.l1._bid = 995
        tob_ask = 1000
        self.l1._update_bid_book(6, tob_ask)
        # case 1: Add orders from 956 -> 995
        for p in range(956, 996):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 40)
    
    def test_update_bid_book2(self):
        ''' 2. _bid < prevailing best ask: add new bid orders from _bid and down '''
        # Create bids from 960 - 990
        for p in range(960, 991):
            self.l1._add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 991):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        # case 2: _bid = 995, best_ask = 1000 -> add 5 new prices: 991 - 995
        self.l1._bid = 995
        tob_ask = 1000
        self.l1._update_bid_book(6, tob_ask)
        for p in range(960, 996):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 5)
        
    def test_update_bid_book3(self):
        ''' _bid >= prevailing best ask: add new bid orders from prevailing best ask-1 and down '''
        # Create bids from 960 - 995
        for p in range(960, 996):
            self.l1._add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 996):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        # case 2: _bid = 1000, but tob_ask = 988 -> add 2 prices: 996, 997
        self.l1._bid = 1000
        tob_ask = 998
        self.l1._update_bid_book(7, tob_ask)
        for p in range(960, 998):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        for p in range(998, 1000):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._bid_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 2)
        
    def test_update_bid_book4(self):
        ''' 4. _bid == current best bid: check for max size and add size if necessary '''
        # case 4: new bid size == 2 -> replenish size to 5 with new add order
        # Create bids from 960 - 995
        for p in range(960, 996):
            self.l1._add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 996):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        self.l1._bid = 995
        tob_ask = 1000
        self.l1._modify_order(Side.BID, 3, 36, 995)
        self.assertEqual(self.l1._bid_book[995]['orders'][36]['quantity'], 2)
        self.assertEqual(self.l1._bid_book[995]['size'], 2)
        self.assertEqual(self.l1._bid_book[995]['num_orders'], 1)
        self.l1.quote_collector.clear() # happens in process_order
        self.l1._update_bid_book(8, tob_ask)
        self.assertEqual(self.l1._bid_book[995]['size'], 5)
        self.assertEqual(self.l1._bid_book[995]['num_orders'], 2)
        self.assertEqual(len(self.l1.quote_collector), 1)
        
    def test_update_bid_book5(self):
        ''' Also, price range should always be between best bid - 20 and best bid - 60 '''
        # make best bid == 975 -> add orders to the other end of the book to make 40 prices
        for p in range(960, 976):
            self.l1._add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 976):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        self.l1._bid = 975
        tob_ask = 1040
        self.l1._update_bid_book(9, tob_ask)
        for p in range(936, 975):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        # make best bid == 1030 -> cancel orders on the other end of the book to make 40 prices
        self.l1._bid = 1030
        tob_ask = 1040
        self.l1._update_bid_book(10, tob_ask)
        for p in range(991, 1031):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        for p in range(961, 991):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._bid_book_prices)
                
    def test_seed_book(self):
        ask = 998
        bid = 990
        step = 20
        self.l1.seed_book(step, ask, bid)
        self.assertEqual(self.l1._mid, 994)
        self.assertTrue(990 in self.l1._bid_book_prices)
        self.assertTrue(998 in self.l1._ask_book_prices)
        self.assertEqual(self.l1._bid, 990)
        self.assertEqual(self.l1._ask, 998)
        self.assertEqual(len(self.l1.quote_collector), 2)

    def test_process_signals(self):
        ''' Test process_signal1 and process_signal2 '''
        signal = {'oibv': 6, 'arrv': 8, 'mid': 1000, 'oib': '011111000000011111000000',
                  'arr': '1222102221222222', 'vol': 4}
        
        self.l1._current_oi_strat = ['221212222222222222020222']
        self.l1._current_arr_strat = '1222102221222222'
        self.l1._current_spradj_strat = ['21220']
        self.l1._last_buy_prices = [998, 999]
        self.l1._last_sell_prices = [1001, 1002]
        self.l1._delta_inv = 3
        
        ask = 1015 # stub quotes?
        bid = 985 # stub quotes?
        step = 20
        
        self.l1.seed_book(step, ask, bid)
        self.assertEqual(self.l1._bid, 985)
        self.assertEqual(self.l1._ask, 1015)
        self.l1.process_signal1(44, signal)
        
        # Step 1: update scores for predictors:
        self.assertListEqual(self.l1._oi_strat['221212222222222222020222']['accuracy'], [9, 1, 9.0])
        self.assertListEqual(self.l1._arr_strat['1222102221222222']['accuracy'], [0, 1, 0.0])
        self.assertListEqual(self.l1._spradj_strat['21220']['rr_spread'], [6, 4, 1.5])
        # Step 2: update the midpoint:
        self.assertEqual(self.l1._mid, 994)
        # Step 3: update spread: using updated spradj_strat = '21220', adjustment == 0 -> bid == 990, ask == 998
        self.assertEqual(self.l1._bid, 990)
        self.assertEqual(self.l1._ask, 998)
        # Step 4: process cancels: there aren't any
        # Step 5: update the book
        tob_bid = 988
        tob_ask = 1000
        self.l1.process_signal2(step, tob_bid, tob_ask)
        for p in range(951, 991):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._bid_book_prices)
        for p in range(998, 1038):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._ask_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 78)
        # Step 6: update cash flow collector, reset inventory, clear recent prices
        self.assertDictEqual(self.l1.cash_flow_collector[-1], {'mmid': 3001, 'timestamp': 20, 'cash_flow': 0, 'delta_inv': 3})
        self.assertFalse(self.l1._delta_inv)
        self.assertFalse(self.l1._last_buy_prices)
        self.assertFalse(self.l1._last_sell_prices)
        
    def test_find_winners(self):
        for j, k in enumerate(self.l1._oi_strat.keys()):
            self.l1._oi_strat[k]['accuracy'][2] = -j
        for j, k in enumerate(self.l1._arr_strat.keys()):
            self.l1._arr_strat[k]['accuracy'][2] = -j
        for j, k in enumerate(self.l1._spradj_strat.keys()):
            self.l1._spradj_strat[k]['rr_spread'][2] = j
        self.l1._find_winners()
        oi_accs = [kv[1]['accuracy'][2] for kv in self.l1._oi_strat.items()]
        for j in range(-99, -19):
            self.assertTrue(j in oi_accs)
        self.assertEqual(min(oi_accs), -99)
        self.assertEqual(max(oi_accs), -20)
        arr_accs = [kv[1]['accuracy'][2] for kv in self.l1._arr_strat.items()]
        for j in range(-99, -19):
            self.assertTrue(j in arr_accs)
        self.assertEqual(min(arr_accs), -99)
        self.assertEqual(max(arr_accs), -20)
        spr_rr = [kv[1]['rr_spread'][2] for kv in self.l1._spradj_strat.items()]
        for j in range(5, 25):
            self.assertTrue(j in spr_rr)
        self.assertEqual(min(spr_rr), 5)
        self.assertEqual(max(spr_rr), 24)
    @unittest.skip('For now')    
    def test_uniform_selection(self):
        for j, k in enumerate(self.l1._oi_strat.keys()):
            self.l1._oi_strat[k]['accuracy'][2] = -j
        for j, k in enumerate(self.l1._arr_strat.keys()):
            self.l1._arr_strat[k]['accuracy'][2] = -j
        for j, k in enumerate(self.l1._spradj_strat.keys()):
            self.l1._spradj_strat[k]['rr_spread'][2] = j
        self.l1._find_winners()
        self.l1._uniform_selection()
    @unittest.skip('For now')    
    def test_weighted_selection(self):
        for j, k in enumerate(self.l1._oi_strat.keys()):
            self.l1._oi_strat[k]['accuracy'][2] = -j
        for j, k in enumerate(self.l1._arr_strat.keys()):
            self.l1._arr_strat[k]['accuracy'][2] = -j
        for j, k in enumerate(self.l1._spradj_strat.keys()):
            self.l1._spradj_strat[k]['rr_spread'][2] = j
        self.l1._find_winners()
        self.l1._weighted_selection()
    
    ''' Test before and after length of strategy dict
    The genetic manipulations are straightforward but run inline, making
    the length of the new strategy dict the only testable outcome
    '''
      
    def test_oi_genes_us(self):
        for j, k in enumerate(self.l1._oi_strat.keys()):
            self.l1._oi_strat[k]['accuracy'][0] = -j
            self.l1._oi_strat[k]['accuracy'][1] = 1
            self.l1._oi_strat[k]['accuracy'][2] = -j
        self.assertEqual(len(self.l1._oi_strat), self.l1._oi_ngene)
        self.l1._find_winners()
        self.assertEqual(len(self.l1._oi_strat), self.l1._oi_keep)
        self.l1._oi_genes_us()
        self.assertEqual(len(self.l1._oi_strat), self.l1._oi_ngene)
        #for k in self.l1._oi_strat.keys():
            #if self.l1._oi_strat[k]['accuracy'][1] != 1:
                #print(self.l1._oi_strat[k])
                
    def test_arr_genes_us(self):
        for j, k in enumerate(self.l1._arr_strat.keys()):
            self.l1._arr_strat[k]['accuracy'][0] = -j
            self.l1._arr_strat[k]['accuracy'][1] = 1
            self.l1._arr_strat[k]['accuracy'][2] = -j
        self.assertEqual(len(self.l1._arr_strat), self.l1._arr_ngene)
        self.l1._find_winners()
        self.assertEqual(len(self.l1._arr_strat), self.l1._arr_keep)
        self.l1._arr_genes_us()
        self.assertEqual(len(self.l1._arr_strat), self.l1._arr_ngene)
        #for k in self.l1._arr_strat.keys():
            #if self.l1._arr_strat[k]['accuracy'][1] != 1:
                #print(self.l1._arr_strat[k])
                
    def test_spr_genes_us(self):
        for j, k in enumerate(self.l1._spradj_strat.keys()):
            self.l1._spradj_strat[k]['rr_spread'][0] = j
            self.l1._spradj_strat[k]['rr_spread'][1] = 1
            self.l1._spradj_strat[k]['rr_spread'][2] = j
        self.assertEqual(len(self.l1._spradj_strat), self.l1._spr_ngene)
        self.l1._find_winners()
        self.assertEqual(len(self.l1._spradj_strat), self.l1._spradj_keep)
        self.l1._spr_genes_us()
        self.l1._find_winners()
        self.l1._spr_genes_us()
        self.assertEqual(len(self.l1._spradj_strat), self.l1._spr_ngene)
        #for k in self.l1._spradj_strat.keys():
            #if self.l1._spradj_strat[k]['rr_spread'][1] != 1:
                #print(self.l1._spradj_strat[k])
                
    def test_oi_genes_ws(self):
        for j, k in enumerate(self.l1._oi_strat.keys()):
            self.l1._oi_strat[k]['accuracy'][0] = -j
            self.l1._oi_strat[k]['accuracy'][1] = 1
            self.l1._oi_strat[k]['accuracy'][2] = -j
        self.assertEqual(len(self.l1._oi_strat), self.l1._oi_ngene)
        self.l1._find_winners()
        self.assertEqual(len(self.l1._oi_strat), self.l1._oi_keep)
        self.l1._oi_genes_ws()
        self.assertEqual(len(self.l1._oi_strat), self.l1._oi_ngene)
        #for k in self.l1._oi_strat.keys():
            #if self.l1._oi_strat[k]['accuracy'][1] != 1:
                #print(self.l1._oi_strat[k])
                
    def test_arr_genes_ws(self):
        for j, k in enumerate(self.l1._arr_strat.keys()):
            self.l1._arr_strat[k]['accuracy'][0] = -j
            self.l1._arr_strat[k]['accuracy'][1] = 1
            self.l1._arr_strat[k]['accuracy'][2] = -j
        self.assertEqual(len(self.l1._arr_strat), self.l1._arr_ngene)
        self.l1._find_winners()
        self.assertEqual(len(self.l1._arr_strat), self.l1._arr_keep)
        self.l1._arr_genes_ws()
        self.assertEqual(len(self.l1._arr_strat), self.l1._arr_ngene)
        #for k in self.l1._arr_strat.keys():
            #if self.l1._arr_strat[k]['accuracy'][1] != 1:
                #print(self.l1._arr_strat[k])
                
    def test_spr_genes_ws(self):
        for j, k in enumerate(self.l1._spradj_strat.keys()):
            self.l1._spradj_strat[k]['rr_spread'][0] = j
            self.l1._spradj_strat[k]['rr_spread'][1] = 1
            self.l1._spradj_strat[k]['rr_spread'][2] = j
        self.assertEqual(len(self.l1._spradj_strat), self.l1._spr_ngene)
        self.l1._find_winners()
        self.assertEqual(len(self.l1._spradj_strat), self.l1._spradj_keep)
        self.l1._spr_genes_ws()
        self.assertEqual(len(self.l1._spradj_strat), self.l1._spr_ngene)
        #for k in self.l1._spradj_strat.keys():
            #if self.l1._spradj_strat[k]['rr_spread'][1] != 1:
                #print(self.l1._spradj_strat[k])