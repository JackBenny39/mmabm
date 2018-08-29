# distutils: language = c++

import random

import numpy as np
cimport numpy as np

from libc.math cimport log
from mmabm.sharedc cimport Side, OType, TType


cdef class ZITrader:
    '''
    ZITrader generates quotes (dicts) based on mechanical probabilities.
    
    A general base class for specific trader types.
    Public attributes: quote_collector
    Public methods: none
    '''
    trader_type = TType.ZITrader

    def __init__(self, int name, int maxq):
        '''
        Initialize ZITrader with some base class attributes and a method
        
        quote_collector is a public container for carrying quotes to the exchange
        '''
        self.trader_id = name # trader id
        self.quantity = self._make_q(maxq)
        self.quote_collector = []
        self._quote_sequence = 0
        
    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2})'.format(class_name, self.trader_id, self.quantity)
    
    def __str__(self):
        return str(tuple([self.trader_id, self.quantity]))
    
    cdef int _make_q(self, int maxq):
        '''Determine order size'''
        cdef np.ndarray default_arr = np.array([1, 5, 10, 25, 50], dtype=np.int)
        return random.choice(default_arr[default_arr<=maxq])
    
    cdef dict _make_add_quote(self, int time, Side side, int price):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        return {'order_id': self._quote_sequence, 'trader_id': self.trader_id, 'timestamp': time, 
                'type': OType.ADD, 'quantity': self.quantity, 'side': side, 'price': price}
        
        
cdef class Provider(ZITrader):
    '''
    Provider generates quotes (dicts) based on make probability.
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector, local_book
    Public methods: confirm_cancel_local, confirm_trade_local, process_signal, bulk_cancel
    '''
    trader_type = TType.Provider
        
    def __init__(self, int name, int maxq, double delta):
        '''Provider has own delta; a local_book to track outstanding orders and a 
        cancel_collector to convey cancel messages to the exchange.
        '''
        super().__init__(name, maxq)
        self._delta = delta
        self.local_book = {}
        self.cancel_collector = []
                
    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3})'.format(class_name, self.trader_id, self.quantity, self._delta)
    
    def __str__(self):
        return str(tuple([self.trader_id, self.quantity, self._delta]))
    
    cdef dict _make_cancel_quote(self, dict q, int time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': q['trader_id'],
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}

    cpdef confirm_trade_local(self, dict confirm):
        to_modify = self.local_book[confirm['order_id']]
        if confirm['quantity'] == to_modify['quantity']:
            del self.local_book[to_modify['order_id']]
        else:
            self.local_book[confirm['order_id']]['quantity'] -= confirm['quantity']
            
    cpdef bulk_cancel(self, int time):
        '''bulk_cancel cancels _delta percent of outstanding orders'''
        self.cancel_collector.clear()
        for x in self.local_book.keys():
            if random.random() < self._delta:
                self.cancel_collector.append(self._make_cancel_quote(self.local_book[x], time))
        for c in self.cancel_collector:        
            del self.local_book[c['order_id']]

    cpdef dict process_signalp(self, int time, dict qsignal, double q_provider, double lambda_t):
        '''Provider buys or sells with probability related to q_provide'''
        cdef int price
        cdef Side side
        cdef dict q
        self.quote_collector.clear()
        if random.random() < q_provider:
            side = Side.BID
            price = self._choose_price_from_exp(side, qsignal['best_ask'], lambda_t)  
        else:
            side = Side.ASK
            price = self._choose_price_from_exp(side, qsignal['best_bid'], lambda_t)
        q = self._make_add_quote(time, side, price)
        self.local_book[q['order_id']] = q
        return q
        
    cdef int _choose_price_from_exp(self, Side side, int inside_price, double lambda_t):
        '''Prices chosen from an exponential distribution'''
        cdef int plug = int(lambda_t*log(random.random()))
        if side == Side.BID:
            return inside_price-1-plug
        else:
            return inside_price+1+plug
    
            
cdef class MarketMaker(Provider):
    '''
    MarketMaker generates a series of quotes near the inside (dicts) based on make probability.
    
    Subclass of Provider
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector (from Provider),
    cash_flow_collector
    Public methods: confirm_cancel_local (from Provider), confirm_trade_local, process_signal 
    '''
    trader_type = TType.MarketMaker

    def __init__(self, int name, int maxq, double delta, int num_quotes, int quote_range):
        '''_num_quotes and _quote_range determine the depth of MM quoting;
        _position and _cashflow are stored MM metrics
        '''
        super().__init__(name, maxq, delta)
        self._num_quotes = num_quotes
        self._quote_range = quote_range
        self._position = 0
        self._cash_flow = 0
        self.cash_flow_collector = []
    
    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3}, {4}, {5})'.format(class_name, self.trader_id, self.quantity, self._delta, self._num_quotes, self._quote_range)
    
    def __str__(self):
        return str(tuple([self.trader_id, self.quantity, self._delta, self._num_quotes, self._quote_range]))
            
    cpdef confirm_trade_local(self, dict confirm):
        '''Modify _cash_flow and _position; update the local_book'''
        if confirm['side'] == Side.BID:
            self._cash_flow -= confirm['price']*confirm['quantity']
            self._position += confirm['quantity']
        else:
            self._cash_flow += confirm['price']*confirm['quantity']
            self._position -= confirm['quantity']
        to_modify = self.local_book[confirm['order_id']]
        if confirm['quantity'] == to_modify['quantity']:
            del self.local_book[to_modify['order_id']]
        else:
            self.local_book[confirm['order_id']]['quantity'] -= confirm['quantity']
        self._cumulate_cashflow(confirm['timestamp'])
         
    cdef void _cumulate_cashflow(self, int timestamp):
        self.cash_flow_collector.append({'mmid': self.trader_id, 'timestamp': timestamp, 'cash_flow': self._cash_flow,
                                         'position': self._position})
            
    cpdef process_signalm(self, int time, dict qsignal, double q_provider):
        '''
        MM chooses prices from a grid determined by the best prevailing prices.
        MM never joins the best price if it has size=1.
        ''' 
        cdef int max_bid_price, min_ask_price, price
        cdef np.ndarray prices
        cdef Side side
        cdef dict q
        self.quote_collector.clear()
        if random.random() < q_provider:
            max_bid_price = qsignal['best_bid'] if qsignal['bid_size'] > 1 else qsignal['best_bid'] - 1
            prices = np.random.choice(range(max_bid_price-self._quote_range+1, max_bid_price+1), size=self._num_quotes)
            side = Side.BID
        else:
            min_ask_price = qsignal['best_ask'] if qsignal['ask_size'] > 1 else qsignal['best_ask'] + 1
            prices = np.random.choice(range(min_ask_price, min_ask_price+self._quote_range), size=self._num_quotes)
            side = Side.ASK
        for price in prices:
            q = self._make_add_quote(time, side, price)
            self.local_book[q['order_id']] = q
            self.quote_collector.append(q)
            

cdef class PennyJumper(ZITrader):
    '''
    PennyJumper jumps in front of best quotes when possible
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector
    Public methods: confirm_trade_local (from ZITrader)
    '''
    trader_type = TType.PennyJumper
    
    def __init__(self, int name, int maxq, int mpi):
        '''
        Initialize PennyJumper
        
        cancel_collector is a public container for carrying cancel messages to the exchange
        PennyJumper tracks private _ask_quote and _bid_quote to determine whether it is alone
        at the inside or not.
        '''
        super().__init__(name, maxq)
        self._mpi = mpi
        self.cancel_collector = []
        self._ask_quote = None
        self._bid_quote = None
    
    def __repr__(self):
        class_name = type(self).__name__
        return '{0}({1}, {2}, {3})'.format(class_name, self.trader_id, self.quantity, self._mpi)

    def __str__(self):
        return str(tuple([self.trader_id, self.quantity, self._mpi]))
    
    cdef dict _make_cancel_quote(self, dict q, int time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': q['trader_id'],
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}

    cpdef confirm_trade_local(self, dict confirm):
        '''PJ has at most one bid and one ask outstanding - if it executes, set price None'''
        if confirm['side'] == Side.BID:
            self._bid_quote = None
        else:
            self._ask_quote = None
            
    cpdef process_signalj(self, int time, dict qsignal, double q_taker):
        '''PJ determines if it is alone at the inside, cancels if not and replaces if there is an available price 
        point inside the current quotes.
        '''
        self.quote_collector.clear()
        self.cancel_collector.clear()
        if qsignal['best_ask'] - qsignal['best_bid'] > self._mpi:
            # q_taker > 0.5 implies greater probability of a buy order; PJ jumps the bid
            if random.random() < q_taker:
                if self._bid_quote: # check if not alone at the bid
                    if self._bid_quote['price'] < qsignal['best_bid'] or self._bid_quote['quantity'] < qsignal['bid_size']:
                        self.cancel_collector.append(self._make_cancel_quote(self._bid_quote, time))
                        self._bid_quote = None
                if not self._bid_quote:
                    self._bid_quote = self._make_add_quote(time, Side.BID, qsignal['best_bid'] + self._mpi)
                    self.quote_collector.append(self._bid_quote)
            else:
                if self._ask_quote: # check if not alone at the ask
                    if self._ask_quote['price'] > qsignal['best_ask'] or self._ask_quote['quantity'] < qsignal['ask_size']:
                        self.cancel_collector.append(self._make_cancel_quote(self._ask_quote, time))
                        self._ask_quote = None
                if not self._ask_quote:
                    self._ask_quote = self._make_add_quote(time, Side.ASK, qsignal['best_ask'] - self._mpi)
                    self.quote_collector.append(self._ask_quote)
        else: # spread = mpi
            if self._bid_quote: # check if not alone at the bid
                if self._bid_quote['price'] < qsignal['best_bid'] or self._bid_quote['quantity'] < qsignal['bid_size']:
                    self.cancel_collector.append(self._make_cancel_quote(self._bid_quote, time))
                    self._bid_quote = None
            if self._ask_quote: # check if not alone at the ask
                if self._ask_quote['price'] > qsignal['best_ask'] or self._ask_quote['quantity'] < qsignal['ask_size']:
                    self.cancel_collector.append(self._make_cancel_quote(self._ask_quote, time))
                    self._ask_quote = None
                              

cdef class Taker(ZITrader):
    '''
    Taker generates quotes (dicts) based on take probability.
        
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal 
    '''
    trader_type = TType.Taker

    def __init__(self, int name, int maxq):
        super().__init__(name, maxq)
        
    cpdef dict process_signalt(self, int time, double q_taker):
        '''Taker buys or sells with 50% probability.'''
        self.quote_collector.clear()
        if random.random() < q_taker: # q_taker > 0.5 implies greater probability of a buy order
            return self._make_add_quote(time, Side.BID, 2000000)
        else:
            return self._make_add_quote(time, Side.ASK, 0)
        
        
cdef class InformedTrader(ZITrader):
    '''
    InformedTrader generates quotes (dicts) based upon a fixed direction
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal
    '''
    trader_type = TType.Informed
    
    def __init__(self, int name, int maxq):
        super().__init__(name, maxq)
        self._side = random.choice([Side.BID, Side.ASK])
        self._price = 0 if self._side == Side.ASK else 2000000
        
    cpdef dict process_signali(self, int time):
        '''InformedTrader buys or sells pre-specified attribute.'''
        self.quote_collector.clear()
        return self._make_add_quote(time, self._side, self._price)
