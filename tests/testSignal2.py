import operator
import unittest

from mmabm.signal2 import OrderSignal, RetSignal, make_oi_signal, make_arr_signal


class TestOrderSignal(unittest.TestCase):
    
    def setUp(self):
        self.oi_inputs = [-16, -8, -6, -4, -2, 0, 0, 2, 4, 6, 8, 16, -8, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 8]
        self.arr_inputs = [0, 1, 2, 4, 8, 16, 32, 64, 0, 1, 2, 3, 4, 6, 8, 12]
        self.hist_len = 5
        self.oi_signal = OrderSignal(self.oi_inputs, self.hist_len)
        self.arr_signal = OrderSignal(self.arr_inputs, self.hist_len)

    def test_setUp(self):
        self.assertEqual(self.oi_signal.v, 0)
        self.assertEqual(len(self.oi_signal.history), self.hist_len)
        self.assertEqual(self.oi_signal._hist_len, self.hist_len)
        self.assertListEqual(self.oi_signal._values, self.oi_inputs)
        self.assertFalse(self.oi_signal.str)

        self.assertEqual(self.arr_signal.v, 0)
        self.assertEqual(len(self.arr_signal.history), self.hist_len)
        self.assertEqual(self.arr_signal._hist_len, self.hist_len)
        self.assertListEqual(self.arr_signal._values, self.arr_inputs)
        self.assertFalse(self.arr_signal.str)

    def test_make_history(self):
        self.oi_signal.v = 1
        self.oi_signal.make_history(1)
        self.assertListEqual(self.oi_signal.history, [0, 1, 0, 0, 0])

        self.arr_signal.v = 2
        self.arr_signal.make_history(2)
        self.assertListEqual(self.arr_signal.history, [0, 0, 2, 0, 0])

    def test_make_part_str(self):
        # oi_signal values for positions 0 - 5: [-16, -8, -6, -4, -2, 0]
        self.assertListEqual(self.oi_signal.make_part_str(-5, operator.le, 0, 6), 
                             ['0', '0','0', '1', '1', '1'])
        self.assertListEqual(self.oi_signal.make_part_str(-7, operator.ge, 0, 6), 
                             ['1', '1', '0', '0', '0', '0'])

    def test_reset_current(self):
        self.assertEqual(self.arr_signal.v, 0)
        self.arr_signal.v = 3
        self.assertEqual(self.arr_signal.v, 3)
        self.arr_signal.reset_current()
        self.assertEqual(self.arr_signal.v, 0)

    def test_make_oi_signal(self):
        self.oi_signal.v = 1
        make_oi_signal(self.oi_signal, 1)
        self.assertEqual(self.oi_signal.v, 1)
        self.assertListEqual(self.oi_signal.history, [0, 1, 0, 0, 0])
        self.assertEqual(self.oi_signal.str, '000000100000000000110000')
        self.oi_signal.v = 3
        make_oi_signal(self.oi_signal, 2)
        self.assertEqual(self.oi_signal.v, 3)
        self.assertListEqual(self.oi_signal.history, [0, 1, 3, 0, 0])
        self.assertEqual(self.oi_signal.str, '000000111000000000111100')
        self.oi_signal.v = 2
        make_oi_signal(self.oi_signal, 3)
        self.assertEqual(self.oi_signal.v, 2)
        self.assertListEqual(self.oi_signal.history, [0, 1, 3, 2, 0])
        self.assertEqual(self.oi_signal.str, '000000111100000000111000')
        self.oi_signal.v = -1
        make_oi_signal(self.oi_signal, 4)
        self.assertEqual(self.oi_signal.v, -1)
        self.assertListEqual(self.oi_signal.history, [0, 1, 3, 2, -1])
        self.assertEqual(self.oi_signal.str, '000000111000000011000000')
        self.oi_signal.v = -4
        make_oi_signal(self.oi_signal, 5)
        self.assertEqual(self.oi_signal.v, -4)
        self.assertListEqual(self.oi_signal.history, [-4, 1, 3, 2, -1])
        self.assertEqual(self.oi_signal.str, '000000100000011111000000')

    def test_make_arr_signal(self):
        make_arr_signal(self.arr_signal, 1)
        print(self.arr_signal.v)
        print(self.arr_signal.history)
        print(self.arr_signal._hist_len)
        print(self.arr_signal._values)
        print(self.arr_signal.str)


class TestRetSignal(unittest.TestCase):
    
    def setUp(self):
        ret_len = 10
        self.ret_signal = RetSignal(ret_len)

    def test_setUp(self):
        print(self.ret_signal.mid)
        print(self.ret_signal.lag_mid)
        print(self.ret_signal.history)
        print(self.ret_signal._hist_len)