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
                self.assertTrue(p in self.l1._local_book.bid_book_prices)
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
                self.assertTrue(p in self.l1._local_book.ask_book_prices)
        for p in range(1005, 1008):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._local_book.ask_book_prices)
        self.assertEqual(len(self.l1.cancel_collector), 3)
        # case 2b: new bid = 987 -> cancel 988, 989, 990
        self.l1._ask = 1000
        self.l1._bid = 987
        self.l1._process_cancels(8)
        for p in range(960, 988):
            with self.subTest(p=p):
                self.assertTrue(p in self.l1._local_book.bid_book_prices)
        for p in range(988, 991):
            with self.subTest(p=p):
                self.assertFalse(p in self.l1._local_book.bid_book_prices)
        self.assertEqual(len(self.l1.cancel_collector), 3)