

class Signal:
    '''
    Class to store, update and manipulate the signal
    '''


    def __init__(self):
        '''
        Constructor
        '''
        self.oibv = 0
        self.oib_str = '0' * 24
        self.oibv5 = [0, 0, 0, 0, 0]
        self.oib_values = [-16, -8, -6, -4, -2, 0, 0, 2, 4, 6, 8, 16, -8, -4, -3, -2, -1, 0, 0, 1, 2, 3, 4, 8]
        
    def make_oib_signal(self, step):
        self.oibv5[step % 5] = self.oibv
        sum5 = sum(self.oibv5)
        oib_list = ['1' if sum5 <= v else '0' for v in self.oib_values[:6]]
        oib_list.extend(['1' if sum5 >= v else '0' for v in self.oib_values[6:12]])
        oib_list.extend(['1' if self.oibv <= v else '0' for v in self.oib_values[12:18]])
        oib_list.extend(['1' if self.oibv >= v else '0' for v in self.oib_values[18:]])
        self.oib_str = ''.join(oib_list)
        
    def reset_current(self):
        self.oibv = 0