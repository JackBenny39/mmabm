import operator
import unittest

from mmabm.signal2 import ImbalanceSignal, RetSignal


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


class TestArrivalSignal(unittest.TestCase):
    '''
    Placeholder for now
    '''
    
    def setUp(self):
        self.arr_inputs = [0, 1, 2, 4, 8, 16, 32, 64, 0, 1, 2, 3, 4, 6, 8, 12]
        self.hist_len = 5
        #self.arr_signal = ImbalanceSignal(self.arr_inputs, self.hist_len)
    @unittest.skip('For now')
    def test_setUp(self):
        #self.assertEqual(self.arr_signal.v, 0)
        #self.assertEqual(len(self.arr_signal.history), self.hist_len)
        #self.assertEqual(self.arr_signal._hist_len, self.hist_len)
        #self.assertListEqual(self.arr_signal._values, self.arr_inputs)
        #self.assertFalse(self.arr_signal.str)
        pass
    @unittest.skip('For now')
    def test_update_v(self):
        self.oi_signal.update_v(7)
        self.assertEqual(self.oi_signal.v, 7)
    @unittest.skip('For now')
    def test_make_history(self):
        #self.arr_signal.v = 2
        #self.arr_signal.make_history(2)
        #self.assertListEqual(self.arr_signal.history, [0, 0, 2, 0, 0])
        pass
    @unittest.skip('For now')
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
    @unittest.skip('For now')
    def test_reset_current(self):
        self.assertEqual(self.oi_signal.v, 0)
        self.oi_signal.v = 3
        self.assertEqual(self.oi_signal.v, 3)
        self.oi_signal.reset_current()
        self.assertEqual(self.oi_signal.v, 0)
    


class TestRetSignal(unittest.TestCase):
    
    def setUp(self):
        ret_len = 10
        self.ret_signal = RetSignal(ret_len)

    def test_setUp(self):
        print(self.ret_signal.mid)
        print(self.ret_signal.lag_mid)
        print(self.ret_signal.history)
        print(self.ret_signal._hist_len)