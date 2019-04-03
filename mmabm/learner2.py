
from mmabm.genetics2 import Predictors
from mmabm.localbook import Localbook
from mmabm.shared import Side, OType, TType

from mmabm.settings import *


class MarketMakerL:
    
    trader_type = TType.MarketMaker
    
    def __init__(self, name, maxq, arrInt, g_int):
        self.trader_id = name # trader id
        self._maxq = maxq
        self.arrInt = arrInt

        self._localbook = Localbook()
        self._quote_sequence = 0

        self.quote_collector = []
        self.cancel_collector = []

        self._oi = Predictors(OI_NUM_CHROMS, OI_COND_LEN, OI_ACTION_LEN, OI_COND_PROBS, 
                              OI_ACTION_MUTATE_P, OI_COND_CROSS_P, OI_COND_MUTATE_P, 
                              OI_THETA, OI_KEEP_PCT, OI_SYMM, OI_WEIGHTS)

        self._genetic_int = g_int
        self.signal_collector = []

    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3})'.format(class_name, self.trader_id, self._maxq, self.arrInt)
    
    def __str__(self):
        return str(tuple([self.trader_id, self._maxq, self.arrInt]))

    def seed_book(self, step, ask, bid):
        q = self._make_add_quote(step, Side.BID, bid, self._maxq)
        self.quote_collector.append(q)
        self._localbook.add_order(q)
        self._bid = bid
        q = self._make_add_quote(step, Side.ASK, ask, self._maxq)
        self.quote_collector.append(q)
        self._localbook.add_order(q)
        self._ask = ask
        self._mid = (ask+bid)/2

    # Make Orders
    def _make_add_quote(self, time, side, price, quantity):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        return {'order_id': self._quote_sequence, 'trader_id': self.trader_id, 'timestamp': time, 
                'type': OType.ADD, 'quantity': quantity, 'side': side, 'price': price}
        
    def _make_cancel_quote(self, q, time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': self.trader_id,
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}

    # Process Signal
    def process_signal1(self, step, signal):
        '''
        The signal is a dict with features of the market state: 
            order imbalance: 24 bits
            
        The midpoint is a function of forecast order flow and inventory imbalance:
            mid(t) = mid(t-1) + D + c*I
            where D is the forecast order imbalance, I is the (change in) inventory imbalance and c is a parameter
        '''

        # Update predictor accuracy
        self._oi.update_accuracies(actual) # need actual