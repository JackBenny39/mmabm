import unittest

from mmabm.signal import Signal


class TestTrader(unittest.TestCase):
    
    def setUp(self):
        self.s1 = Signal()
        
    def test_make_oib_signal(self):
        self.s1.oibv5 = [-2, -4, 5, 0, -3]
        self.s1.oibv = -4
        step = 24
        print(self.s1.oib_str)
        self.s1.make_oib_signal(step)
        print(self.s1.oib_str)
        
    def test_make_arr_signal(self):
        self.s1.arrv5 = [1, 3, 5, 0, 7]
        self.s1.arrv = 2
        step = 24
        print(self.s1.arr_str)
        self.s1.make_arr_signal(step)
        print(self.s1.arr_str)
        
    def test_make_mid_signal(self):
        self.s1.ret10 = [1, .5, -.2, .4, -1, .2, .6, -.2, -.4, -.7]
        self.s1.midl1 = 100
        ask = 104
        bid = 102
        step = 25
        print(self.s1.ret10)
        self.s1.make_mid_signal(step, bid, ask)
        print(self.s1.ret10)
        print(self.s1.mid, self.s1.midl1)
        
    def test_make_vol_signal(self):
        self.s1.ret10 = [1, .5, -.2, .4, -1, .2, .6, -.2, -.4, -.7]
        print(self.s1.make_vol_signal())