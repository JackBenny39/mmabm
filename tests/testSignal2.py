import operator
import unittest

from mmabm.signal2 import ImbalanceSignal, OrderFlowSignal, RetSignal


class TestImbalanceSignal(unittest.TestCase):
    
    def setUp(self):
        self.oi_inputs = [-16, -8, -6, -4, -2, 0, 0, 2, 4, 6, 8, 16, -8, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 8]
        self.hist_len = 5
        self.oi_signal = ImbalanceSignal(self.oi_inputs, self.hist_len)

    def test_setUp(self):
        self.assertEqual(self.oi_signal.v, 0)
        self.assertEqual(len(self.oi_signal._history), self.hist_len)
        self.assertEqual(self.oi_signal._hist_len, self.hist_len)
        self.assertListEqual(self.oi_signal._values, self.oi_inputs)
        self.assertFalse(self.oi_signal.str)

    def test_update_v(self):
        self.oi_signal.update_v(7)
        self.assertEqual(self.oi_signal.v, 7)

    def test_make_history(self):
        self.oi_signal.v = 1
        self.oi_signal._make_history(1)
        self.assertListEqual(self.oi_signal._history, [0, 1, 0, 0, 0])

    def test_make_signal(self):
        self.oi_signal.v = 1
        self.oi_signal.make_signal(1)
        self.assertEqual(self.oi_signal.v, 1)
        self.assertListEqual(self.oi_signal._history, [0, 1, 0, 0, 0])
        self.assertEqual(self.oi_signal.str, '000000100000000000110000')
        self.oi_signal.v = 3
        self.oi_signal.make_signal(2)
        self.assertEqual(self.oi_signal.v, 3)
        self.assertListEqual(self.oi_signal._history, [0, 1, 3, 0, 0])
        self.assertEqual(self.oi_signal.str, '000000111000000000111100')
        self.oi_signal.v = 2
        self.oi_signal.make_signal(3)
        self.assertEqual(self.oi_signal.v, 2)
        self.assertListEqual(self.oi_signal._history, [0, 1, 3, 2, 0])
        self.assertEqual(self.oi_signal.str, '000000111100000000111000')
        self.oi_signal.v = -1
        self.oi_signal.make_signal(4)
        self.assertEqual(self.oi_signal.v, -1)
        self.assertListEqual(self.oi_signal._history, [0, 1, 3, 2, -1])
        self.assertEqual(self.oi_signal.str, '000000111000000011000000')
        self.oi_signal.v = -4
        self.oi_signal.make_signal(5)
        self.assertEqual(self.oi_signal.v, -4)
        self.assertListEqual(self.oi_signal._history, [-4, 1, 3, 2, -1])
        self.assertEqual(self.oi_signal.str, '000000100000011111000000')

    def test_reset_current(self):
        self.assertEqual(self.oi_signal.v, 0)
        self.oi_signal.v = 3
        self.assertEqual(self.oi_signal.v, 3)
        self.oi_signal.reset_current()
        self.assertEqual(self.oi_signal.v, 0)


class TestOrderFlowSignal(unittest.TestCase):
    '''
    Placeholder for now
    '''
    
    def setUp(self):
        self.of_inputs = [0, 1, 2, 4, 8, 16, 32, 64, 0, 1, 2, 3, 4, 6, 8, 12]
        self.hist_len = 5
        self.of_signal = OrderFlowSignal(self.of_inputs, self.hist_len)

    def test_setUp(self):
        self.assertEqual(self.of_signal.v, 0)
        self.assertEqual(len(self.of_signal._history), self.hist_len)
        self.assertEqual(self.of_signal._hist_len, self.hist_len)
        self.assertListEqual(self.of_signal._values, self.of_inputs)
        self.assertFalse(self.of_signal.str)

    def test_update_v(self):
        self.of_signal.update_v(7)
        self.assertEqual(self.of_signal.v, 7)

    def test_make_history(self):
        self.of_signal.v = 2
        self.of_signal._make_history(2)
        self.assertListEqual(self.of_signal._history, [0, 0, 2, 0, 0])

    def test_make_signal(self):
        self.of_signal.v = 1
        self.of_signal.make_signal(1)
        self.assertEqual(self.of_signal.v, 1)
        self.assertListEqual(self.of_signal._history, [0, 1, 0, 0, 0])
        self.assertEqual(self.of_signal.str, '1000000010000000')
        self.of_signal.v = 3
        self.of_signal.make_signal(2)
        self.assertEqual(self.of_signal.v, 3)
        self.assertListEqual(self.of_signal._history, [0, 1, 3, 0, 0])
        self.assertEqual(self.of_signal.str, '1110000011100000')
        self.of_signal.v = 2
        self.of_signal.make_signal(3)
        self.assertEqual(self.of_signal.v, 2)
        self.assertListEqual(self.of_signal._history, [0, 1, 3, 2, 0])
        self.assertEqual(self.of_signal.str, '1111000011000000')
        self.of_signal.v = 8
        self.of_signal.make_signal(4)
        self.assertEqual(self.of_signal.v, 8)
        self.assertListEqual(self.of_signal._history, [0, 1, 3, 2, 8])
        self.assertEqual(self.of_signal.str, '1111100011111100')
        self.of_signal.v = 4
        self.of_signal.make_signal(5)
        self.assertEqual(self.of_signal.v, 4)
        self.assertListEqual(self.of_signal._history, [4, 1, 3, 2, 8])
        self.assertEqual(self.of_signal.str, '1111110011110000')

    def test_reset_current(self):
        self.assertEqual(self.of_signal.v, 0)
        self.of_signal.v = 3
        self.assertEqual(self.of_signal.v, 3)
        self.of_signal.reset_current()
        self.assertEqual(self.of_signal.v, 0)
    


class TestRetSignal(unittest.TestCase):
    
    def setUp(self):
        ret_len = 10
        self.ret_signal = RetSignal(ret_len)

    def test_setUp(self):
        print(self.ret_signal.mid)
        print(self.ret_signal.lag_mid)
        print(self.ret_signal.history)
        print(self.ret_signal._hist_len)