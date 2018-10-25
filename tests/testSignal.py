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