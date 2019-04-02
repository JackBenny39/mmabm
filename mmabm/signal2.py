import operator

from math import log
from statistics import pstdev


class OrderSignal:
    '''
    Orders-based Signals
    '''

    def __init__(self, inputs, hist_len):
        '''
        Constructor
        '''
        self.v = 0
        self.history = [0] * hist_len
        self._hist_len = hist_len
        self._values = inputs
        self.str = None

    def make_history(self, step):
        self.history[step % self._hist_len] = self.v

    def make_part_str(self, val, op, start, stop):
        return ['1' if op(val, v) else '0' for v in self._values[start:stop]]
    
    def reset_current(self):
        self.v = 0


class RetSignal:
    '''
    Return Signal
    '''

    def __init__(self, hist_len):
        '''
        Constructor
        '''
        self.mid = 0
        self.lag_mid = 0
        self.history = [0] * hist_len
        self._hist_len = hist_len

    def make_history(self, step, bid, ask):
        self.mid = (ask + bid) / 2
        self.history[step % self._hist_len] = 100 * log(self.mid/self.lag_mid)
        self.lag_mid = self.mid

    def make_volatility(self):
        return pstdev(self.history)


def make_oi_signal(signal, step):
    signal.make_history(step)
    sum_hist = sum(signal.history)
    oi_list = signal.make_part_str(sum_hist, operator.le, 0, 6)
    oi_list.extend(signal.make_part_str(sum_hist, operator.ge, 6, 12))
    oi_list.extend(signal.make_part_str(signal.v, operator.le, 12, 18))
    oi_list.extend(signal.make_part_str(signal.v, operator.ge, 18, 24))
    signal.str = ''.join(oi_list)

def make_arr_signal(signal, step):
    signal.make_history(step)
    sum_hist = sum(signal.history)
    arr_list = signal.make_part_str(sum_hist, operator.ge, 0, 8)
    arr_list.extend(signal.make_part_str(signal.v, operator.ge, 8, 16))
    signal.str = ''.join(arr_list)