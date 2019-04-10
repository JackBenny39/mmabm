from mmabm.localbook import Localbook
from mmabm.shared import Side, OType
import unittest


class TestLocalbook(unittest.TestCase):
    
    def setUp(self):
        '''
        setUp creates the Localbook instance and a set of orders
        '''
        self.local = Localbook()
        self.q1_buy = {'order_id': 1, 'trader_id': 1001,'timestamp': 2, 'type': OType.ADD, 
                       'quantity': 1, 'side': Side.BID, 'price': 50}
        self.q2_buy = {'order_id': 2, 'trader_id': 1001, 'timestamp': 3, 'type': OType.ADD, 
                       'quantity': 1, 'side': Side.BID, 'price': 50}
        self.q3_buy = {'order_id': 1, 'trader_id': 1010, 'timestamp': 4, 'type': OType.ADD, 
                       'quantity': 3, 'side': Side.BID, 'price': 49}
        self.q4_buy = {'order_id': 1, 'trader_id': 1011, 'timestamp': 5, 'type': OType.ADD, 
                       'quantity': 3, 'side': Side.BID, 'price': 47}
        self.q1_sell = {'order_id': 3, 'trader_id': 1001, 'timestamp': 2, 'type': OType.ADD, 
                        'quantity': 1, 'side': Side.ASK, 'price': 52}
        self.q2_sell = {'order_id': 4, 'trader_id': 1001, 'timestamp': 3, 'type': OType.ADD, 
                        'quantity': 1, 'side': Side.ASK, 'price': 52}
        self.q3_sell = {'order_id': 2, 'trader_id': 1010, 'timestamp': 4, 'type': OType.ADD, 
                        'quantity': 3, 'side': Side.ASK, 'price': 53}
        self.q4_sell = {'order_id': 2, 'trader_id': 1011, 'timestamp': 5, 'type': OType.ADD, 
                        'quantity': 3, 'side': Side.ASK, 'price': 55}

    def test_add_order(self):
        # 2 buy orders
        self.assertFalse(self.local.bid_book_prices)
        self.assertFalse(self.local.bid_book)
        self.local.add_order(self.q1_buy)
        self.assertTrue(50 in self.local.bid_book_prices)
        self.assertTrue(50 in self.local.bid_book.keys())
        self.assertEqual(self.local.bid_book[50]['num_orders'], 1)
        self.assertEqual(self.local.bid_book[50]['size'], 1)
        self.assertEqual(self.local.bid_book[50]['order_ids'][0], 1)
        del self.q1_buy['type']
        del self.q1_buy['trader_id']
        self.assertDictEqual(self.local.bid_book[50]['orders'][1], self.q1_buy)
        self.local.add_order(self.q2_buy)
        self.assertEqual(self.local.bid_book[50]['num_orders'], 2)
        self.assertEqual(self.local.bid_book[50]['size'], 2)
        self.assertEqual(self.local.bid_book[50]['order_ids'][1], 2)
        del self.q2_buy['type']
        del self.q2_buy['trader_id']
        self.assertDictEqual(self.local.bid_book[50]['orders'][2], self.q2_buy)
        # 2 sell orders
        self.assertFalse(self.local.ask_book_prices)
        self.assertFalse(self.local.ask_book)
        self.local.add_order(self.q1_sell)
        self.assertTrue(52 in self.local.ask_book_prices)
        self.assertTrue(52 in self.local.ask_book.keys())
        self.assertEqual(self.local.ask_book[52]['num_orders'], 1)
        self.assertEqual(self.local.ask_book[52]['size'], 1)
        self.assertEqual(self.local.ask_book[52]['order_ids'][0], 3)
        del self.q1_sell['type']
        del self.q1_sell['trader_id']
        self.assertDictEqual(self.local.ask_book[52]['orders'][3], self.q1_sell)
        self.local.add_order(self.q2_sell)
        self.assertEqual(self.local.ask_book[52]['num_orders'], 2)
        self.assertEqual(self.local.ask_book[52]['size'], 2)
        self.assertEqual(self.local.ask_book[52]['order_ids'][1], 4)
        del self.q2_sell['type']
        del self.q2_sell['trader_id']
        self.assertDictEqual(self.local.ask_book[52]['orders'][4], self.q2_sell)

    def test_remove_order(self):
        # buy orders
        self.local.add_order(self.q1_buy)
        self.local.add_order(self.q2_buy)
        self.assertTrue(50 in self.local.bid_book_prices)
        self.assertTrue(50 in self.local.bid_book.keys())
        self.assertEqual(self.local.bid_book[50]['num_orders'], 2)
        self.assertEqual(self.local.bid_book[50]['size'], 2)
        self.assertEqual(len(self.local.bid_book[50]['order_ids']), 2)
        # remove first order
        self.local.remove_order(Side.BID, 50, 1)
        self.assertEqual(self.local.bid_book[50]['num_orders'], 1)
        self.assertEqual(self.local.bid_book[50]['size'], 1)
        self.assertEqual(len(self.local.bid_book[50]['order_ids']), 1)
        self.assertFalse(1 in self.local.bid_book[50]['orders'].keys())
        self.assertTrue(50 in self.local.bid_book_prices)
        # remove second order
        self.local.remove_order(Side.BID, 50, 2)
        self.assertFalse(self.local.bid_book_prices)
        self.assertEqual(self.local.bid_book[50]['num_orders'], 0)
        self.assertEqual(self.local.bid_book[50]['size'], 0)
        self.assertEqual(len(self.local.bid_book[50]['order_ids']), 0)
        self.assertFalse(2 in self.local.bid_book[50]['orders'].keys())
        self.assertFalse(50 in self.local.bid_book_prices)
        # remove second order again
        self.local.remove_order(Side.BID, 50, 2)
        self.assertFalse(self.local.bid_book_prices)
        self.assertEqual(self.local.bid_book[50]['num_orders'], 0)
        self.assertEqual(self.local.bid_book[50]['size'], 0)
        self.assertEqual(len(self.local.bid_book[50]['order_ids']), 0)
        self.assertFalse(2 in self.local.bid_book[50]['orders'].keys())
        # sell orders
        self.local.add_order(self.q1_sell)
        self.local.add_order(self.q2_sell)
        self.assertTrue(52 in self.local.ask_book_prices)
        self.assertTrue(52 in self.local.ask_book.keys())
        self.assertEqual(self.local.ask_book[52]['num_orders'], 2)
        self.assertEqual(self.local.ask_book[52]['size'], 2)
        self.assertEqual(len(self.local.ask_book[52]['order_ids']), 2)
        # remove first order
        self.local.remove_order(Side.ASK, 52, 3)
        self.assertEqual(self.local.ask_book[52]['num_orders'], 1)
        self.assertEqual(self.local.ask_book[52]['size'], 1)
        self.assertEqual(len(self.local.ask_book[52]['order_ids']), 1)
        self.assertFalse(3 in self.local.ask_book[52]['orders'].keys())
        self.assertTrue(52 in self.local.ask_book_prices)
        # remove second order
        self.local.remove_order(Side.ASK, 52, 4)
        self.assertFalse(self.local.ask_book_prices)
        self.assertEqual(self.local.ask_book[52]['num_orders'], 0)
        self.assertEqual(self.local.ask_book[52]['size'], 0)
        self.assertEqual(len(self.local.ask_book[52]['order_ids']), 0)
        self.assertFalse(4 in self.local.ask_book[52]['orders'].keys())
        self.assertFalse(52 in self.local.ask_book_prices)
        # remove second order again
        self.local.remove_order(Side.ASK, 52, 4)
        self.assertFalse(self.local.ask_book_prices)
        self.assertEqual(self.local.ask_book[52]['num_orders'], 0)
        self.assertEqual(self.local.ask_book[52]['size'], 0)
        self.assertEqual(len(self.local.ask_book[52]['order_ids']), 0)
        self.assertFalse(4 in self.local.ask_book[52]['orders'].keys())

    def test_modify_order(self):
        # Buy order
        q1 = {'order_id': 1, 'trader_id': 1001, 'timestamp': 5, 'type': OType.ADD,
              'quantity': 2, 'side': Side.BID, 'price': 50}
        self.local.add_order(q1)
        self.assertEqual(self.local.bid_book[50]['size'], 2)
        # remove 1
        self.local.modify_order(Side.BID, 1, 1, 50)
        self.assertEqual(self.local.bid_book[50]['size'], 1)
        self.assertEqual(self.local.bid_book[50]['orders'][1]['quantity'], 1)
        self.assertTrue(self.local.bid_book_prices)
        # remove remainder
        self.local.modify_order(Side.BID, 1, 1, 50)
        self.assertFalse(self.local.bid_book_prices)
        self.assertEqual(self.local.bid_book[50]['num_orders'], 0)
        self.assertEqual(self.local.bid_book[50]['size'], 0)
        self.assertFalse(1 in self.local.bid_book[50]['orders'].keys())
        # Sell order
        q2 = {'order_id': 2, 'trader_id': 1001, 'timestamp': 5, 'type': OType.ADD,
              'quantity': 2, 'side': Side.ASK, 'price': 50}
        self.local.add_order(q2)
        self.assertEqual(self.local.ask_book[50]['size'], 2)
        # remove 1
        self.local.modify_order(Side.ASK, 1, 2, 50)
        self.assertEqual(self.local.ask_book[50]['size'], 1)
        self.assertEqual(self.local.ask_book[50]['orders'][2]['quantity'], 1)
        self.assertTrue(self.local.ask_book_prices)
        # remove remainder
        self.local.modify_order(Side.ASK, 1, 2, 50)
        self.assertFalse(self.local.ask_book_prices)
        self.assertEqual(self.local.ask_book[50]['num_orders'], 0)
        self.assertEqual(self.local.ask_book[50]['size'], 0)
        self.assertFalse(2 in self.local.ask_book[50]['orders'].keys())
        