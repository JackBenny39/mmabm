import unittest

from mmabm.signal2 import OrderSignal, RetSignal, make_oi_signal, make_arr_signal


class TestOrderSignal(unittest.TestCase):
    
    def setUp(self):
        oi_inputs = [-16, -8, -6, -4, -2, 0, 0, 2, 4, 6, 8, 16, -8, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 8]
        arr_inputs = [0, 1, 2, 4, 8, 16, 32, 64, 0, 1, 2, 3, 4, 6, 8, 12]
        hist_len = 5
        self.oi_signal = OrderSignal(oi_inputs, hist_len)
        self.arr_signal = OrderSignal(arr_inputs, hist_len)

    def test_setUp(self):
        print(self.oi_signal.v)
        print(self.oi_signal.history)
        print(self.oi_signal._hist_len)
        print(self.oi_signal._values)
        print(self.oi_signal.str)

        print(self.arr_signal.v)
        print(self.arr_signal.history)
        print(self.arr_signal._hist_len)
        print(self.arr_signal._values)
        print(self.arr_signal.str)

    def test_make_oi_signal(self):
        make_oi_signal(self.oi_signal, 1)
        print(self.oi_signal.v)
        print(self.oi_signal.history)
        print(self.oi_signal._hist_len)
        print(self.oi_signal._values)
        print(self.oi_signal.str)

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