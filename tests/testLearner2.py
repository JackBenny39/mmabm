import random
import unittest

#import numpy as np

from mmabm.shared import Side, OType

from mmabm.learner2 import MarketMakerL


class TestMarketMakerL(unittest.TestCase):
    
    def setUp(self):
        self.l1 = MarketMakerL(3001, 5, 1, 250)

    def test_setUp(self):
        print(self.l1)
        self.assertEqual(self.l1.trader_id, 3001)
        self.assertEqual(self.l1._maxq, 5)
        self.assertEqual(self.l1.arrInt, 1)
        self.assertTrue(self.l1._localbook)
        self.assertFalse(self.l1._quote_sequence)

        self.assertFalse(self.l1.quote_collector)
        self.assertFalse(self.l1.cancel_collector)

        self.assertFalse(self.l1._bid)
        self.assertFalse(self.l1._ask)
        self.assertFalse(self.l1._mid)
        self.assertFalse(self.l1._delta_inv)
        self.assertFalse(self.l1._cash_flow)
        self.assertFalse(self.l1.cash_flow_collector)

        self.assertTrue(self.l1._oi)

        self.assertEqual(self.l1._genetic_int, 250)
        self.assertFalse(self.l1.signal_collector)

    def test_make_add_quote(self):
        self.assertDictEqual(self.l1._make_add_quote(1, Side.BID, 999999, 5),
                             {'order_id': 1, 'trader_id': 3001, 'timestamp': 1,
                              'type': OType.ADD, 'quantity': 5, 'side': Side.BID,
                              'price': 999999})

    def test_make_cancel_quote(self):
        q = {'order_id': 2, 'quantity': 5, 'side': Side.BID, 'price': 999998}
        self.assertDictEqual(self.l1._make_cancel_quote(q, 3),
                             {'type': OType.CANCEL, 'timestamp': 3, 'order_id': 2, 'trader_id': 3001,
                              'quantity': 5, 'side': Side.BID, 'price': 999998})

    def test_seed_book(self):
        self.l1.seed_book(5, 1000002, 999998)
        self.assertEqual(len(self.l1.quote_collector), 2)
        self.assertEqual(self.l1._bid, 999998)
        self.assertEqual(self.l1._ask, 1000002)
        self.assertEqual(self.l1._mid, 1000000)

    def test_confirm_trade_local(self):
        # Add orders to localbook
        q1 = {'order_id': 1, 'trader_id': 3001,'timestamp': 2, 'type': OType.ADD, 
              'quantity': 7, 'side': Side.BID, 'price': 999998}
        q2 = {'order_id': 2, 'trader_id': 3001,'timestamp': 2, 'type': OType.ADD, 
              'quantity': 7, 'side': Side.ASK, 'price': 1000002}
        self.l1._localbook.add_order(q1)
        self.l1._localbook.add_order(q2)
        # Confirm trades on both sides
        c1 = {'price': 999998, 'side': Side.BID, 'quantity': 5, 'order_id': 1}
        self.l1.confirm_trade_local(c1)
        self.assertEqual(self.l1._cash_flow, -49.9999)
        self.assertEqual(self.l1._delta_inv, 5)
        c2 = {'price': 1000002, 'side': Side.ASK, 'quantity': 5, 'order_id': 2}
        self.l1.confirm_trade_local(c2)
        self.assertAlmostEqual(self.l1._cash_flow, 0.0002)
        self.assertEqual(self.l1._delta_inv, 0)

    def test_cumulate_cashflow(self):
        self.l1._cash_flow = 100
        self.l1._delta_inv = -5
        self.l1.cumulate_cashflow(7)
        self.assertDictEqual(self.l1.cash_flow_collector[0], {'mmid': 3001, 'timestamp': 7,
                             'cash_flow': 100, 'delta_inv': -5})

    def test_update_midpoint(self):
        self.l1._update_midpoint(999, 1001)
        self.assertEqual(self.l1._mid, 1000)

    def test_make_spread(self):
        random.seed(39) # randint(-2, 2) generates -1, then 0
        self.l1._make_spread(995, 1005)
        self.assertEqual(self.l1._bid, 995)
        self.assertEqual(self.l1._ask, 1004)
        # randint(-2, 2) generates 1, then -2 -> use a locked market
        # step 1 of _make spread yields 998 - 997
        # first call to random() decrements the bid by 1: 997 - 997
        # second call to random() decrements the bid by 1: 996 - 997
        self.l1._make_spread(1000, 996)
        self.assertEqual(self.l1._bid, 996)
        self.assertEqual(self.l1._ask, 997)

    def test_collect_signal(self):
        step = 7
        signal = (-5, '100101')
        self.assertFalse(self.l1.signal_collector)
        self.assertFalse(self.l1._oi.current)
        null_p = self.l1._oi.predictors[0]
        self.l1._oi.current.append(null_p)
        self.l1._collect_signal(step, signal)
        keep = {'Step': 7, 'OIV': -5, 'OIStr': '100101', 'OICond': null_p.condition, 
                'OIStrat': null_p.strategy, 'OIAcc': null_p.accuracy}
        self.assertDictEqual(self.l1.signal_collector[0], keep)
        self.assertEqual(len(self.l1.signal_collector), 1)

    def test_process_cancels(self):
        # Create asks from 1005 - 1035
        for p in range(1005, 1036):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(1005, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        # Create bids from 960 - 990
        for p in range(960, 991):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 991):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
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
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        for p in range(1005, 1008):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._localbook.ask_book_prices)
        self.assertEqual(len(self.l1.cancel_collector), 3)
        # case 2b: new bid = 987 -> cancel 988, 989, 990
        self.l1._ask = 1000
        self.l1._bid = 987
        self.l1._process_cancels(8)
        for p in range(960, 988):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        for p in range(988, 991):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._localbook.bid_book_prices)
        self.assertEqual(len(self.l1.cancel_collector), 3)

    ''' Several cases:
    1. _ask > prevailing worst ask (ask book empty due to canceling first)
    2. _ask > prevailing best bid: add new ask orders from _ask and up
    3. _ask <= prevailing best bid: add new ask orders from prevailing best bid+1 and up
    4. _ask == current best ask: check for max size and add size if necessary
    Also, price range should always be between best ask + 20 and best ask + 60
    '''
    def test_update_ask_book1(self):
        # 1. _ask > prevailing worst ask (ask book empty due to canceling first)
        self.assertFalse(self.l1._localbook.ask_book_prices)
        self.l1._ask = 1000
        tob_bid = 995
        self.l1._update_ask_book(6, tob_bid)
        # case 1: Add orders from 1000 -> 1039
        for p in range(1000, 1040):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 40)

    def test_update_ask_book2(self):
        ''' 2. _ask > prevailing best bid: add new ask orders from _ask and up '''
        # Create asks from 1005 - 1035
        for p in range(1005, 1036):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(1005, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        # case 2: _ask = 1000, best_bid = 995 -> add 5 new prices: 1000 - 1004
        self.l1._ask = 1000
        tob_bid = 995
        self.l1._update_ask_book(6, tob_bid)
        for p in range(1000, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 5)
 
    def test_update_ask_book3(self):
        ''' 3. _ask <= prevailing best bid: add new ask orders from prevailing best bid+1 and up '''
        for p in range(1000, 1036):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(1000, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        # case 3: _ask = 990 but best_bid = 995 -> add 4 new prices: 996 - 999
        self.l1._ask = 990
        tob_bid = 995
        self.l1._update_ask_book(7, tob_bid)
        for p in range(996, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        for p in range(990, 996):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._localbook.ask_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 4)
 
    def test_update_ask_book4(self):
        ''' 4. _ask == current best ask: check for max size and add size if necessary '''
        for p in range(996, 1036):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(996, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        # case 4: new ask size == 2 -> replenish size to 5 with new add order
        self.l1._ask = 996
        tob_bid = 990
        self.l1._localbook.modify_order(Side.ASK, 3, 1, 996)
        self.assertEqual(self.l1._localbook.ask_book[996]['orders'][1]['quantity'], 2)
        self.assertEqual(self.l1._localbook.ask_book[996]['size'], 2)
        self.assertEqual(self.l1._localbook.ask_book[996]['num_orders'], 1)
        self.l1.quote_collector.clear() # happens in process_order
        self.l1._update_ask_book(8, tob_bid)
        self.assertEqual(self.l1._localbook.ask_book[996]['size'], 5)
        self.assertEqual(self.l1._localbook.ask_book[996]['num_orders'], 2)
        self.assertEqual(len(self.l1.quote_collector), 1)
  
    def test_update_ask_book5(self):
        ''' Also, price range should always be between best ask + 20 and best ask + 60 '''
        # make best ask == 1020 -> add orders to the other end of the book to make 40 prices
        for p in range(1020, 1036):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.ASK, p, self.l1._maxq))
        for p in range(1020, 1036):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        tob_bid = 990
        self.l1._ask = 1020
        self.l1._update_ask_book(10, tob_bid)
        for p in range(1020, 1059):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        # make best ask == 980 -> cancel orders on the other end of the book to make 40 prices
        self.l1._bid = 975
        self.l1._ask = 980
        tob_bid = 975
        self.l1._update_ask_book(10, tob_bid)
        for p in range(980, 1019):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.ask_book_prices)
        for p in range(1020, 1059):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._localbook.ask_book_prices)
                
    ''' Several cases:
    1. _bid < prevailing worst bid (bid book empty due to canceling first)
    2. _bid < prevailing best ask: add new bid orders from _bid and down
    3. _bid >= prevailing best ask: add new bid orders from prevailing best ask-1 and down
    4. _bid == current best bid: check for max size and add size if necessary
    Also, price range should always be between best bid - 20 and best bid - 60
    '''
    def test_update_bid_book1(self):
        ''' 1. _bid < prevailing worst bid (bid book empty due to canceling first) '''
        self.assertFalse(self.l1._localbook.bid_book_prices)
        self.l1._bid = 995
        tob_ask = 1000
        self.l1._update_bid_book(6, tob_ask)
        # case 1: Add orders from 956 -> 995
        for p in range(956, 996):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 40)

    def test_update_bid_book2(self):
        ''' 2. _bid < prevailing best ask: add new bid orders from _bid and down '''
        # Create bids from 960 - 990
        for p in range(960, 991):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 991):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        # case 2: _bid = 995, best_ask = 1000 -> add 5 new prices: 991 - 995
        self.l1._bid = 995
        tob_ask = 1000
        self.l1._update_bid_book(6, tob_ask)
        for p in range(960, 996):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 5)
   
    def test_update_bid_book3(self):
        ''' _bid >= prevailing best ask: add new bid orders from prevailing best ask-1 and down '''
        # Create bids from 960 - 995
        for p in range(960, 996):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 996):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        # case 2: _bid = 1000, but tob_ask = 988 -> add 2 prices: 996, 997
        self.l1._bid = 1000
        tob_ask = 998
        self.l1._update_bid_book(7, tob_ask)
        for p in range(960, 998):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        for p in range(998, 1000):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._localbook.bid_book_prices)
        self.assertEqual(len(self.l1.quote_collector), 2)
 
    def test_update_bid_book4(self):
        ''' 4. _bid == current best bid: check for max size and add size if necessary '''
        # case 4: new bid size == 2 -> replenish size to 5 with new add order
        # Create bids from 960 - 995
        for p in range(960, 996):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 996):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        self.l1._bid = 995
        tob_ask = 1000
        self.l1._localbook.modify_order(Side.BID, 3, 36, 995)
        self.assertEqual(self.l1._localbook.bid_book[995]['orders'][36]['quantity'], 2)
        self.assertEqual(self.l1._localbook.bid_book[995]['size'], 2)
        self.assertEqual(self.l1._localbook.bid_book[995]['num_orders'], 1)
        self.l1.quote_collector.clear() # happens in process_order
        self.l1._update_bid_book(8, tob_ask)
        self.assertEqual(self.l1._localbook.bid_book[995]['size'], 5)
        self.assertEqual(self.l1._localbook.bid_book[995]['num_orders'], 2)
        self.assertEqual(len(self.l1.quote_collector), 1)
  
    def test_update_bid_book5(self):
        ''' Also, price range should always be between best bid - 20 and best bid - 60 '''
        # make best bid == 975 -> add orders to the other end of the book to make 40 prices
        for p in range(960, 976):
            self.l1._localbook.add_order(self.l1._make_add_quote(35, Side.BID, p, self.l1._maxq))
        for p in range(960, 976):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        self.l1._bid = 975
        tob_ask = 1040
        self.l1._update_bid_book(9, tob_ask)
        for p in range(936, 975):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        # make best bid == 1030 -> cancel orders on the other end of the book to make 40 prices
        self.l1._bid = 1030
        tob_ask = 1040
        self.l1._update_bid_book(10, tob_ask)
        for p in range(991, 1031):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._localbook.bid_book_prices)
        for p in range(961, 991):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._localbook.bid_book_prices)