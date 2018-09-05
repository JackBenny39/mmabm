import random
import unittest

import numpy as np

from mmabm.shared import Side, OType
from mmabm.trader import ZITrader, Provider, MarketMaker, PennyJumper, Taker, InformedTrader, MarketMakerL


class TestTrader(unittest.TestCase):
    
    def setUp(self):
        self.z1 = ZITrader(1, 5)
        self.p1 = Provider(1001, 1, 0.025)
        self.l1 = self._makeMML(3001)
        self.m1 = MarketMaker(3001, 1, 0.05, 12, 60)
        self.j1 = PennyJumper(4001, 1, 5)
        self.t1 = Taker(2001, 1)
        self.i1 = InformedTrader(5001, 1)
        
        self.q1 = {'order_id': 1, 'timestamp': 1, 'type': OType.ADD, 'quantity': 1, 'side': Side.BID,
                   'price': 125}
        self.q2 = {'order_id': 2, 'timestamp': 2, 'type': OType.ADD, 'quantity': 5, 'side': Side.BID,
                   'price': 125}
        self.q3 = {'order_id': 3, 'timestamp': 3, 'type': OType.ADD, 'quantity': 1, 'side': Side.BID,
                   'price': 124}
        self.q4 = {'order_id': 4, 'timestamp': 4, 'type': OType.ADD, 'quantity': 1, 'side': Side.BID,
                   'price': 123}
        self.q5 = {'order_id': 5, 'timestamp': 5, 'type': OType.ADD, 'quantity': 1, 'side': Side.BID,
                   'price': 122}
        self.q6 = {'order_id': 6, 'timestamp': 6, 'type': OType.ADD, 'quantity': 1, 'side': Side.ASK,
                   'price': 126}
        self.q7 = {'order_id': 7, 'timestamp': 7, 'type': OType.ADD, 'quantity': 5, 'side': Side.ASK,
                   'price': 127}
        self.q8 = {'order_id': 8, 'timestamp': 8, 'type': OType.ADD, 'quantity': 1, 'side': Side.ASK,
                   'price': 128}
        self.q9 = {'order_id': 9, 'timestamp': 9, 'type': OType.ADD, 'quantity': 1, 'side': Side.ASK,
                   'price': 129}
        self.q10 = {'order_id': 10, 'timestamp': 10, 'type': OType.ADD, 'quantity': 1, 'side': Side.ASK,
                   'price': 130}
    
    def _makeMML(self, tid):
        bit_n = 5
        gene_n = 21
        genes = tuple([''.join(str(x) for x in [random.randint(0, 1) for _ in range(bit_n)]) for _ in range(gene_n)])
        return MarketMakerL(tid, genes)
        
# ZITrader tests

    def test_repr_ZITrader(self):
        self.assertEqual('ZITrader({0}, {1})'.format(self.z1.trader_id, self.z1.quantity), '{0!r}'.format(self.z1))
   
    def test_str_ZITrader(self):
        self.assertEqual('({0!r}, {1})'.format(self.z1.trader_id, self.z1.quantity), '{0}'.format(self.z1))

    def test_make_q(self):
        self.assertLessEqual(self.z1.quantity, 5)

    def test_make_add_quote(self):
        time = 1
        side = Side.ASK
        price = 125
        q = self.z1._make_add_quote(time, side, price)
        expected = {'order_id': 1, 'trader_id': self.z1.trader_id, 'timestamp': 1, 'type': OType.ADD, 
                    'quantity': self.z1.quantity, 'side': Side.ASK, 'price': 125}
        self.assertDictEqual(q, expected)
        
# Provider tests  

    def test_repr_Provider(self):
        self.assertEqual('Provider({0}, {1}, {2})'.format(self.p1.trader_id, self.p1.quantity, self.p1._delta),
                         '{0!r}'.format(self.p1))
   
    def test_str__Provider(self):
        self.assertEqual('({0!r}, {1}, {2})'.format(self.p1.trader_id, self.p1.quantity, self.p1._delta), 
                         '{0}'.format(self.p1))
        
    def test_make_cancel_quote_Provider(self):
        self.q1['trader_id'] = self.p1.trader_id
        q = self.p1._make_cancel_quote(self.q1, 2)
        expected = {'order_id': 1, 'trader_id': self.p1.trader_id, 'timestamp': 2, 'type': OType.CANCEL, 
                    'quantity': 1, 'side': Side.BID, 'price': 125}
        self.assertDictEqual(q, expected)

    def test_confirm_trade_local_Provider(self):
        '''
        Test Provider for full and partial trade
        '''
        # Provider
        self.q1['trader_id'] = self.p1.trader_id
        self.q2['trader_id'] = self.p1.trader_id
        self.p1.local_book[self.q1['order_id']] = self.q1
        self.p1.local_book[self.q2['order_id']] = self.q2
        # trade full quantity of q1
        trade1 = {'timestamp': 2, 'trader_id': self.p1.trader_id, 'order_id': 1, 'quantity': 1, 'side': Side.BID, 'price': 2000000}
        self.assertEqual(len(self.p1.local_book), 2)
        self.p1.confirm_trade_local(trade1)
        self.assertEqual(len(self.p1.local_book), 1)
        expected = {self.q2['order_id']: self.q2}
        self.assertDictEqual(self.p1.local_book, expected)
        # trade partial quantity of q2
        trade2 = {'timestamp': 3, 'trader_id': self.p1.trader_id, 'order_id': 2, 'quantity': 2, 'side': Side.BID, 'price': 2000000}
        self.p1.confirm_trade_local(trade2)
        self.assertEqual(len(self.p1.local_book), 1)
        expected = {'order_id': 2, 'timestamp': 2, 'trader_id': self.p1.trader_id, 'type': OType.ADD, 'quantity': 3, 
                    'side': Side.BID, 'price': 125}
        self.assertDictEqual(self.p1.local_book.get(trade2['order_id']), expected) 
    
    def test_choose_price_from_exp(self):
        sell_price = self.p1._choose_price_from_exp(Side.BID, 75000, -100)
        self.assertLess(sell_price, 75000)
        buy_price = self.p1._choose_price_from_exp(Side.ASK, 25000, -100)
        self.assertGreater(buy_price, 25000)
        self.assertEqual(np.remainder(buy_price,1),0)
        self.assertEqual(np.remainder(sell_price,1),0)
    
    def test_process_signal_Provider(self):
        time = 1
        q_provider = 0.5
        tob_price = {'best_bid': 25000, 'best_ask': 75000}
        self.assertFalse(self.p1.local_book)
        random.seed(1)
        q1 = self.p1.process_signal(time, tob_price, q_provider, -100)
        self.assertEqual(q1['side'], Side.BID)
        self.assertEqual(len(self.p1.local_book), 1)
        q2 = self.p1.process_signal(time, tob_price, q_provider, -100)
        self.assertEqual(q2['side'], Side.ASK)
        self.assertEqual(len(self.p1.local_book), 2)
        
    def test_bulk_cancel_Provider(self):
        '''
        Put 10 orders in the book, use random seed to determine which orders are cancelled,
        test for cancelled orders in the queue
        '''
        self.assertFalse(self.p1.local_book)
        self.assertFalse(self.p1.cancel_collector)
        self.q1['trader_id'] = self.p1.trader_id
        self.q2['trader_id'] = self.p1.trader_id
        self.q3['trader_id'] = self.p1.trader_id
        self.q4['trader_id'] = self.p1.trader_id
        self.q5['trader_id'] = self.p1.trader_id
        self.q6['trader_id'] = self.p1.trader_id
        self.q7['trader_id'] = self.p1.trader_id
        self.q8['trader_id'] = self.p1.trader_id
        self.q9['trader_id'] = self.p1.trader_id
        self.q10['trader_id'] = self.p1.trader_id
        self.p1.local_book[self.q1['order_id']] = self.q1
        self.p1.local_book[self.q2['order_id']] = self.q2
        self.p1.local_book[self.q3['order_id']] = self.q3
        self.p1.local_book[self.q4['order_id']] = self.q4
        self.p1.local_book[self.q5['order_id']] = self.q5
        self.p1.local_book[self.q6['order_id']] = self.q6
        self.p1.local_book[self.q7['order_id']] = self.q7
        self.p1.local_book[self.q8['order_id']] = self.q8
        self.p1.local_book[self.q9['order_id']] = self.q9
        self.p1.local_book[self.q10['order_id']] = self.q10
        self.assertEqual(len(self.p1.local_book), 10)
        self.assertFalse(self.p1.cancel_collector)
        # random seed = 1 generates 1 position less than 0.03 from random.random: 9
        random.seed(1)
        self.p1._delta = 0.03
        self.p1.bulk_cancel(11)
        self.assertEqual(len(self.p1.cancel_collector), 1)
        # random seed = 7 generates 3 positions less than 0.1 from random.random: 3, 6, 9
        random.seed(7)
        self.p1._delta = 0.1
        self.p1.bulk_cancel(12)
        self.assertEqual(len(self.p1.cancel_collector), 3)
        # random seed = 39 generates 0 position less than 0.1 from random.random
        random.seed(39)
        self.p1._delta = 0.1
        self.p1.bulk_cancel(12)
        self.assertFalse(self.p1.cancel_collector)
        
        
# MarketMakerL tests
        
    def test_make_strategy(self):
        expected = {-10: -3, -9: -2, -8: 8, -7: -7, -6: 9, -5: 10, -4: -9, -3: -1, -2: -3,
                    -1: -4, 0: -3, 1: 0, 2: 6, 3: -8, 4: 10, 5: 14, 6: -4, 7: 14, 8: -9,
                    9: 7, 10: 3}
        self.assertDictEqual(expected, self.l1._strategy)
        
    def test_make_add_quote_MML(self):
        time = 1
        side = Side.ASK
        price = 125
        quantity = 5
        q = self.l1._make_add_quote(time, side, price, quantity)
        expected = {'order_id': 1, 'trader_id': self.l1.trader_id, 'timestamp': 1, 'type': OType.ADD, 
                    'quantity': quantity, 'side': Side.ASK, 'price': 125}
        self.assertDictEqual(q, expected)
        
    def test_make_cancel_quote_MML(self):
        self.q1['trader_id'] = self.l1.trader_id
        q = self.l1._make_cancel_quote(self.q1, 2)
        expected = {'order_id': 1, 'trader_id': self.l1.trader_id, 'timestamp': 2, 'type': OType.CANCEL, 
                    'quantity': 1, 'side': Side.BID, 'price': 125}
        self.assertDictEqual(q, expected)
   
    def test_process_signal_MML(self):
        signal = {'net_of': 1, 'delta_mid': 0}
        time = 10
        tob1 = {'best_bid': 25000, 'best_ask': 75000, 'bid_size': 10, 'ask_size': 10}
        temp_mid = self.l1.process_signal(time, tob1, signal)
        delta = self.l1._strategy[signal['net_of']]
        self.assertEqual(50000 + delta, temp_mid)
        print(temp_mid)

   
        
# MarketMaker tests
   
    def test_repr_MarketMaker(self):
        self.assertEqual('MarketMaker({0}, {1}, {2}, {3}, {4})'.format(self.m1.trader_id, self.m1.quantity, self.m1._delta, 
                                                                       self.m1._num_quotes, self.m1._quote_range), '{0!r}'.format(self.m1))
   
    def test_str_MarketMaker(self):
        self.assertEqual('({0!r}, {1}, {2}, {3}, {4})'.format(self.m1.trader_id, self.m1.quantity, self.m1._delta, 
                                                              self.m1._num_quotes, self.m1._quote_range), '{0}'.format(self.m1))

    def test_confirm_trade_local_MM(self):
        '''
        Test Market Maker for full and partial trade
        '''
        # MarketMaker buys
        self.q1['trader_id'] = self.m1.trader_id
        self.q2['trader_id'] = self.m1.trader_id
        self.m1.local_book[self.q1['order_id']] = self.q1
        self.m1.local_book[self.q2['order_id']] = self.q2
        # trade full quantity of q1
        trade1 = {'timestamp': 2, 'trader_id': self.p1.trader_id, 'order_id': 1, 'quantity': 1, 'side': Side.BID, 'price': 2000000}
        self.assertEqual(len(self.m1.local_book), 2)
        self.m1.confirm_trade_local(trade1)
        self.assertEqual(len(self.m1.local_book), 1)
        self.assertEqual(self.m1._position, 1)
        expected = {self.q2['order_id']: self.q2}
        self.assertDictEqual(self.m1.local_book, expected)
        # trade partial quantity of q2
        trade2 = {'timestamp': 3, 'trader_id': self.p1.trader_id, 'order_id': 2, 'quantity': 2, 'side': Side.BID, 'price': 2000000}
        self.m1.confirm_trade_local(trade2)
        self.assertEqual(len(self.m1.local_book), 1)
        self.assertEqual(self.m1._position, 3)
        expected = {'order_id': 2, 'timestamp': 2, 'trader_id': self.m1.trader_id, 'type': OType.ADD, 'quantity': 3, 
                    'side': Side.BID, 'price': 125}
        self.assertDictEqual(self.m1.local_book.get(trade2['order_id']), expected) 
        
        # MarketMaker sells
        self.setUp()
        self.q6['trader_id'] = self.m1.trader_id
        self.q7['trader_id'] = self.m1.trader_id
        self.m1.local_book[self.q6['order_id']] = self.q6
        self.m1.local_book[self.q7['order_id']] = self.q7
        # trade full quantity of q6
        trade1 = {'timestamp': 6, 'trader_id': self.p1.trader_id, 'order_id': 6, 'quantity': 1, 'side': Side.ASK, 'price': 0}
        self.assertEqual(len(self.m1.local_book), 2)
        self.m1.confirm_trade_local(trade1)
        self.assertEqual(len(self.m1.local_book), 1)
        self.assertEqual(self.m1._position, -1)
        expected = {self.q7['order_id']: self.q7}
        self.assertDictEqual(self.m1.local_book, expected)
        # trade partial quantity of q7
        trade2 = {'timestamp': 7, 'trader_id': self.p1.trader_id, 'order_id': 7, 'quantity': 2, 'side': Side.ASK, 'price': 0}
        self.m1.confirm_trade_local(trade2)
        self.assertEqual(len(self.m1.local_book), 1)
        self.assertEqual(self.m1._position, -3)
        expected = {'order_id': 7, 'timestamp': 7, 'trader_id': self.m1.trader_id, 'type': OType.ADD, 'quantity': 3, 
                    'side': Side.ASK, 'price': 127}
        self.assertDictEqual(self.m1.local_book.get(trade2['order_id']), expected) 
   
    def test_cumulate_cashflow_MM(self):
        self.assertFalse(self.m1.cash_flow_collector)
        expected = {'mmid': 3001, 'timestamp': 10, 'cash_flow': 0, 'position': 0}
        self.m1._cumulate_cashflow(10)
        self.assertDictEqual(self.m1.cash_flow_collector[0], expected)
   
    def test_process_signal_MM1_12(self):
        time = 1
        q_provider = 0.5
        low_ru_seed = 1
        hi_ru_seed = 10
        # size > 1: market maker matches best price
        tob1 = {'best_bid': 25000, 'best_ask': 75000, 'bid_size': 10, 'ask_size': 10}
        self.assertFalse(self.m1.quote_collector)
        self.assertFalse(self.m1.local_book)
        random.seed(low_ru_seed)
        self.m1.process_signal(time, tob1, q_provider)
        self.assertEqual(len(self.m1.quote_collector), 12)
        self.assertEqual(self.m1.quote_collector[0]['side'], Side.BID)
        for i in range(len(self.m1.quote_collector)):
            with self.subTest(i=i):
                self.assertLessEqual(self.m1.quote_collector[i]['price'], 25000)
                self.assertGreaterEqual(self.m1.quote_collector[i]['price'], 24941)
                self.assertTrue(self.m1.quote_collector[i]['price'] in range(24941, 25001))
        self.assertEqual(len(self.m1.local_book), 12)
        random.seed(hi_ru_seed)
        self.m1.process_signal(time, tob1, q_provider)
        self.assertEqual(len(self.m1.quote_collector), 12)
        self.assertEqual(self.m1.quote_collector[0]['side'], Side.ASK)
        for i in range(len(self.m1.quote_collector)):
            with self.subTest(i=i):
                self.assertLessEqual(self.m1.quote_collector[i]['price'], 75060)
                self.assertGreaterEqual(self.m1.quote_collector[i]['price'], 75000)
                self.assertTrue(self.m1.quote_collector[i]['price'] in range(75000, 75061))
        self.assertEqual(len(self.m1.local_book), 24)
        # size == 1: market maker adds liquidity one point behind
        self.setUp()
        tob2 = {'best_bid': 25000, 'best_ask': 75000, 'bid_size': 1, 'ask_size': 1}
        self.assertFalse(self.m1.quote_collector)
        self.assertFalse(self.m1.local_book)
        random.seed(low_ru_seed)
        self.m1.process_signal(time, tob2, q_provider)
        self.assertEqual(len(self.m1.quote_collector), 12)
        self.assertEqual(self.m1.quote_collector[0]['side'], Side.BID)
        for i in range(len(self.m1.quote_collector)):
            with self.subTest(i=i):
                self.assertLessEqual(self.m1.quote_collector[i]['price'], 24999)
                self.assertGreaterEqual(self.m1.quote_collector[i]['price'], 24940)
                self.assertTrue(self.m1.quote_collector[i]['price'] in range(24940, 25000))
        self.assertEqual(len(self.m1.local_book), 12)
        random.seed(hi_ru_seed)
        self.m1.process_signal(time, tob2, q_provider)
        self.assertEqual(len(self.m1.quote_collector), 12)
        self.assertEqual(self.m1.quote_collector[0]['side'], Side.ASK)
        for i in range(len(self.m1.quote_collector)):
            with self.subTest(i=i):
                self.assertLessEqual(self.m1.quote_collector[i]['price'], 75060)
                self.assertGreaterEqual(self.m1.quote_collector[i]['price'], 75001)
                self.assertTrue(self.m1.quote_collector[i]['price'] in range(75001, 75061))
        self.assertEqual(len(self.m1.local_book), 24)
        
# PennyJumper tests

    def test_repr_PennyJumper(self):
        self.assertEqual('PennyJumper({0}, {1}, {2})'.format(self.j1.trader_id, self.j1.quantity, self.j1._mpi), '{0!r}'.format(self.j1))

    def test_str_PennyJumper(self):
        self.assertEqual('({0!r}, {1}, {2})'.format(self.j1.trader_id, self.j1.quantity, self.j1._mpi), '{0}'.format(self.j1))
  
    def test_confirm_trade_local_PJ(self):
        # PennyJumper book
        self.j1._bid_quote = {'order_id': 1, 'trader_id': 4001, 'timestamp': 1, 'type': OType.ADD, 'quantity': 1, 
                              'side': Side.BID, 'price': 125}
        self.j1._ask_quote = {'order_id': 6, 'trader_id': 4001, 'timestamp': 6, 'type': OType.ADD, 'quantity': 1, 
                              'side': Side.ASK, 'price': 126}
        # trade at the bid
        trade1 = {'timestamp': 2, 'trader_id': 4001, 'order_id': 1, 'quantity': 1, 'side': Side.BID, 'price': 0}
        self.assertTrue(self.j1._bid_quote)
        self.j1.confirm_trade_local(trade1)
        self.assertFalse(self.j1._bid_quote)
        # trade at the ask
        trade2 = {'timestamp': 12, 'trader_id': 4001, 'order_id': 6, 'quantity': 1, 'side': Side.ASK, 'price': 2000000}
        self.assertTrue(self.j1._ask_quote)
        self.j1.confirm_trade_local(trade2)
        self.assertFalse(self.j1._ask_quote)
   
    def test_process_signal_PJ(self):
        # spread > mpi
        tob = {'bid_size': 5, 'best_bid': 999990, 'best_ask': 1000005, 'ask_size': 5}
        # PJ book empty
        self.j1._bid_quote = None
        self.j1._ask_quote = None
        # random.seed = 1 generates random.uniform(0,1) = 0.13 then .85
        # jump the bid by 1, then jump the ask by 1
        random.seed(1)
        self.j1.process_signal(5, tob, 0.5)
        self.assertDictEqual(self.j1._bid_quote, {'order_id': 1, 'trader_id': 4001, 'timestamp': 5, 'type': OType.ADD, 
                                                  'quantity': 1, 'side': Side.BID, 'price': 999995})
        tob = {'bid_size': 1, 'best_bid': 999995, 'best_ask': 1000005, 'ask_size': 5}
        self.j1.process_signal(6, tob, 0.5)
        self.assertDictEqual(self.j1._ask_quote, {'order_id': 2, 'trader_id': 4001, 'timestamp': 6, 'type':  OType.ADD, 
                                                  'quantity': 1, 'side': Side.ASK, 'price': 1000000})
        # PJ alone at tob
        tob = {'bid_size': 1, 'best_bid': 999995, 'best_ask': 1000000, 'ask_size': 1}
        # nothing happens
        self.j1.process_signal(7, tob, 0.5)
        self.assertDictEqual(self.j1._bid_quote, {'order_id': 1, 'trader_id': 4001, 'timestamp': 5, 'type':  OType.ADD, 
                                                  'quantity': 1, 'side': Side.BID, 'price': 999995})
        self.assertDictEqual(self.j1._ask_quote, {'order_id': 2, 'trader_id': 4001, 'timestamp': 6, 'type':  OType.ADD, 
                                                  'quantity': 1, 'side': Side.ASK, 'price': 1000000})
        # PJ bid and ask behind the book
        tob = {'bid_size': 1, 'best_bid': 999990, 'best_ask': 1000005, 'ask_size': 1}
        self.j1._bid_quote = {'order_id': 1, 'trader_id': 4001, 'timestamp': 5, 'type':  OType.ADD, 'quantity': 1, 
                              'side': Side.BID, 'price': 999985}
        self.j1._ask_quote = {'order_id': 2, 'trader_id': 4001, 'timestamp': 6, 'type':  OType.ADD, 'quantity': 1, 
                              'side': Side.ASK, 'price': 1000010}
        # random.seed = 1 generates random.uniform(0,1) = 0.13 then .85
        # jump the bid by 1, then jump the ask by 1; cancel old quotes
        random.seed(1)
        self.j1.process_signal(10, tob, 0.5)
        self.assertDictEqual(self.j1._bid_quote, {'order_id': 3, 'trader_id': 4001, 'timestamp': 10, 'type':  OType.ADD, 
                                                  'quantity': 1, 'side': Side.BID, 'price': 999995})
        self.assertDictEqual(self.j1.cancel_collector[0], {'order_id': 1, 'trader_id': 4001, 'timestamp': 10, 'type': OType.CANCEL, 
                                                           'quantity': 1, 'side': Side.BID, 'price': 999985})
        self.assertDictEqual(self.j1.quote_collector[0], self.j1._bid_quote)
        self.j1.process_signal(11, tob, 0.5)
        self.assertDictEqual(self.j1._ask_quote, {'order_id': 4, 'trader_id': 4001, 'timestamp': 11, 'type':  OType.ADD, 
                                                  'quantity': 1, 'side': Side.ASK, 'price': 1000000})
        self.assertDictEqual(self.j1.cancel_collector[0], {'order_id': 2, 'trader_id': 4001, 'timestamp': 11, 'type': OType.CANCEL, 
                                                           'quantity': 1, 'side': Side.ASK, 'price': 1000010})
        self.assertDictEqual(self.j1.quote_collector[0],self.j1._ask_quote)
        # PJ not alone at the inside
        tob = {'bid_size': 5, 'best_bid': 999990, 'best_ask': 1000010, 'ask_size': 5}
        self.j1._bid_quote = {'order_id': 1, 'trader_id': 4001, 'timestamp': 5, 'type':  OType.ADD, 'quantity': 1, 
                              'side': Side.BID, 'price': 999990}
        self.j1._ask_quote = {'order_id': 2, 'trader_id': 4001, 'timestamp': 6, 'type':  OType.ADD, 'quantity': 1, 
                              'side': Side.ASK, 'price': 1000010}
        # random.seed = 1 generates random.uniform(0,1) = 0.13 then .85
        # jump the bid by 1, then jump the ask by 1; cancel old quotes
        random.seed(1)
        self.j1.process_signal(12, tob, 0.5)
        self.assertDictEqual(self.j1._bid_quote, {'order_id': 5, 'trader_id': 4001, 'timestamp': 12, 'type':  OType.ADD, 
                                                  'quantity': 1, 'side': Side.BID, 'price': 999995})
        self.assertDictEqual(self.j1.cancel_collector[0], {'order_id': 1, 'trader_id': 4001, 'timestamp': 12, 'type': OType.CANCEL, 
                                                           'quantity': 1, 'side': Side.BID, 'price': 999990})
        self.assertDictEqual(self.j1.quote_collector[0], self.j1._bid_quote)
        self.j1.process_signal(13, tob, 0.5)
        self.assertDictEqual(self.j1._ask_quote, {'order_id': 6, 'trader_id': 4001, 'timestamp': 13, 'type':  OType.ADD, 
                                                  'quantity': 1, 'side': Side.ASK, 'price': 1000005})
        self.assertDictEqual(self.j1.cancel_collector[0], {'order_id': 2, 'trader_id': 4001, 'timestamp': 13, 'type': OType.CANCEL, 
                                                           'quantity': 1, 'side': Side.ASK, 'price': 1000010})
        self.assertDictEqual(self.j1.quote_collector[0],self.j1._ask_quote)
        # spread at mpi, PJ alone at nbbo
        tob = {'bid_size': 1, 'best_bid': 999995, 'best_ask': 1000000, 'ask_size': 1}
        self.j1._bid_quote = {'order_id': 1, 'trader_id': 4001, 'timestamp': 5, 'type':  OType.ADD, 'quantity': 1, 
                              'side': Side.BID, 'price': 999995}
        self.j1._ask_quote = {'order_id': 2, 'trader_id': 4001, 'timestamp': 6, 'type':  OType.ADD, 'quantity': 1, 
                              'side': Side.ASK, 'price': 1000000}
        random.seed(1)
        self.j1.process_signal(14, tob, 0.5)
        self.assertDictEqual(self.j1._bid_quote, {'order_id': 1, 'trader_id': 4001, 'timestamp': 5, 'type':  OType.ADD, 
                                                  'quantity': 1, 'side': Side.BID, 'price': 999995})
        self.assertFalse(self.j1.cancel_collector)
        self.assertFalse(self.j1.quote_collector)
        self.j1.process_signal(15, tob, 0.5)
        self.assertDictEqual(self.j1._ask_quote, {'order_id': 2, 'trader_id': 4001, 'timestamp': 6, 'type':  OType.ADD, 
                                                  'quantity': 1, 'side': Side.ASK, 'price': 1000000})
        self.assertFalse(self.j1.cancel_collector)
        self.assertFalse(self.j1.quote_collector)
        # PJ bid and ask behind the book
        self.j1._bid_quote = {'order_id': 1, 'trader_id': 4001, 'timestamp': 5, 'type':  OType.ADD, 'quantity': 1, 
                              'side': Side.BID, 'price': 999990}
        self.j1._ask_quote = {'order_id': 2, 'trader_id': 4001, 'timestamp': 6, 'type':  OType.ADD, 'quantity': 1, 
                              'side': Side.ASK, 'price': 1000010}
        # random.seed = 1 generates random.uniform(0,1) = 0.13 then .85
        # cancel bid and ask
        random.seed(1)
        self.assertTrue(self.j1._bid_quote)
        self.assertTrue(self.j1._ask_quote)
        self.j1.process_signal(16, tob, 0.5)
        self.assertFalse(self.j1._bid_quote)
        self.assertFalse(self.j1._ask_quote)
        self.assertDictEqual(self.j1.cancel_collector[0], {'order_id': 1, 'trader_id': 4001, 'timestamp': 16, 'type': OType.CANCEL, 
                                                           'quantity': 1, 'side': Side.BID, 'price': 999990})
        self.assertDictEqual(self.j1.cancel_collector[1], {'order_id': 2, 'trader_id': 4001, 'timestamp': 16, 'type': OType.CANCEL, 
                                                           'quantity': 1, 'side': Side.ASK, 'price': 1000010})
        self.assertFalse(self.j1.quote_collector)
        
# Taker tests
   
    def test_repr_Taker(self):
        self.assertEqual('Taker({0}, {1})'.format(self.t1.trader_id, self.t1.quantity), '{0!r}'.format(self.t1))
   
    def test_str_Taker(self):
        self.assertEqual('({0!r}, {1})'.format(self.t1.trader_id, self.t1.quantity), '{0}'.format(self.t1))
   
    def test_process_signal_Taker(self):
        '''
        Generates a quote object (dict) and appends to quote_collector
        '''
        time = 1
        q_taker = 0.5
        low_ru_seed = 1
        hi_ru_seed = 10
        random.seed(low_ru_seed)
        q1 = self.t1.process_signal(time, q_taker)
        self.assertEqual(q1['side'], Side.BID)
        self.assertEqual(q1['price'], 2000000)
        random.seed(hi_ru_seed)
        q2 = self.t1.process_signal(time, q_taker)
        self.assertEqual(q2['side'], Side.ASK)
        self.assertEqual(q2['price'], 0)
        
# InformedTrader tests
   
    def test_repr_InformedTrader(self):
        self.assertEqual('InformedTrader({0}, {1})'.format(self.i1.trader_id, self.i1.quantity), '{0!r}'.format(self.i1))
   
    def test_str_InformedTrader(self):
        self.assertEqual('({0!r}, {1})'.format(self.i1.trader_id, self.i1.quantity), '{0}'.format(self.i1))
   
    def test_process_signal_InformedTrader(self):
        '''
        Generates a quote object (dict) and appends to quote_collector
        '''
        time1 = 1
        q1 = self.i1.process_signal(time1)
        self.assertEqual(q1['side'], self.i1._side)
        self.assertEqual(q1['price'], self.i1._price)
        self.assertEqual(q1['quantity'], 1)
