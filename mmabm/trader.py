import random

import numpy as np

from math import log
from mmabm.shared import Side, OType, TType


class ZITrader:
    '''
    ZITrader generates quotes (dicts) based on mechanical probabilities.
    
    A general base class for specific trader types.
    Public attributes: quote_collector
    Public methods: none
    '''
    trader_type = TType.ZITrader

    def __init__(self, name, maxq):
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
    
    def _make_q(self, maxq):
        '''Determine order size'''
        default_arr = np.array([1, 5, 10, 25, 50])
        return random.choice(default_arr[default_arr<=maxq])
    
    def _make_add_quote(self, time, side, price):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        return {'order_id': self._quote_sequence, 'trader_id': self.trader_id, 'timestamp': time, 
                'type': OType.ADD, 'quantity': self.quantity, 'side': side, 'price': price}
        
        
class Provider(ZITrader):
    '''
    Provider generates quotes (dicts) based on make probability.
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector, local_book
    Public methods: confirm_cancel_local, confirm_trade_local, process_signal, bulk_cancel
    '''
    trader_type = TType.Provider
        
    def __init__(self, name, maxq, delta):
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
    
    def _make_cancel_quote(self, q, time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': q['trader_id'],
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}

    def confirm_trade_local(self, confirm):
        to_modify = self.local_book[confirm['order_id']]
        if confirm['quantity'] == to_modify['quantity']:
            del self.local_book[to_modify['order_id']]
        else:
            self.local_book[confirm['order_id']]['quantity'] -= confirm['quantity']
            
    def bulk_cancel(self, time):
        '''bulk_cancel cancels _delta percent of outstanding orders'''
        self.cancel_collector.clear()
        for x in self.local_book.keys():
            if random.random() < self._delta:
                self.cancel_collector.append(self._make_cancel_quote(self.local_book[x], time))
        for c in self.cancel_collector:        
            del self.local_book[c['order_id']]

    def process_signal(self, time, qsignal, q_provider, lambda_t):
        '''Provider buys or sells with probability related to q_provide'''
        if random.random() < q_provider:
            side = Side.BID
            price = self._choose_price_from_exp(side, qsignal['best_ask'], lambda_t)
        else:
            side = Side.ASK
            price = self._choose_price_from_exp(side, qsignal['best_bid'], lambda_t)
        q = self._make_add_quote(time, side, price)
        self.local_book[q['order_id']] = q
        return q            
      
    def _choose_price_from_exp(self, side, inside_price, lambda_t):
        '''Prices chosen from an exponential distribution'''
        # make pricing explicit for now. Logic scales for other mpi.
        plug = int(lambda_t*log(random.random()))
        if side == Side.BID:
            return inside_price-1-plug
        else:
            return inside_price+1+plug
    
            
class MarketMaker(Provider):
    '''
    MarketMaker generates a series of quotes near the inside (dicts) based on make probability.
    
    Subclass of Provider
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector (from Provider),
    cash_flow_collector
    Public methods: confirm_cancel_local (from Provider), confirm_trade_local, process_signal 
    '''
    trader_type = TType.MarketMaker

    def __init__(self, name, maxq, delta, num_quotes, quote_range):
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
            
    def confirm_trade_local(self, confirm):
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
         
    def _cumulate_cashflow(self, timestamp):
        self.cash_flow_collector.append({'mmid': self.trader_id, 'timestamp': timestamp, 'cash_flow': self._cash_flow,
                                         'position': self._position})
            
    def process_signal(self, time, qsignal, q_provider):
        '''
        MM chooses prices from a grid determined by the best prevailing prices.
        MM never joins the best price if it has size=1.
        ''' 
        # make pricing explicit for now. Logic scales for other mpi and quote ranges.
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
            
            
class MarketMakerL():
    '''
    MarketMakerL learns from its environment
    
    Environment: order flow, absolute order flow (imbalance), current ask, bid and depths
    What the MML does: prepares ideal order book; compares to actual; add/cancel to make ideal == actual.
    MML chooses: quote midpoint; spread; price range for bids and asks; depth at those prices.
    Midpoint: function of signed order imbalance
    Spread: function of absolute order imbalance
    Depth: function of absolute order imbalance
    Price range: emergent outcome
    
    Public attributes:
    Public methods:
    Private attributes:
    Private methods:
    '''
    trader_type = TType.MarketMaker
    
    def __init__(self, name, a, b, c, geneset):
        self.trader_id = name # trader id
        self._a = a
        self._b = b
        self._c = c
        self._bid_book = {}
        self._bid_book_prices = []
        self._ask_book = {}
        self._ask_book_prices = []
        self._position = []
        self.quote_collector = []
        self.cancel_collector = []
        self._quote_sequence = 0
        
        self._oi_strat, self._oi_len = self._make_oi_strat2(geneset[0])
        self._arr_strat, self._arr_len = self._make_arr_strat2(geneset[1])
        self._askadj_strat, self._ask_len = self._make_bidask_strat2(geneset[2])
        self._bidadj_strat, self._bid_len = self._make_bidask_strat2(geneset[3])


    ''' Old Strategy '''
    def _make_oi_strat(self, oi_chroms):
        oi_strat = {'chromosomes': oi_chroms}
        oi_strat['strategy'] = {k: int(v[1:], 2)*(1 if int(v[0]) else -1) for k, v in oi_chroms.items()}
        oi_strat['accuracy'] = {k: 0 for k in oi_chroms.keys()}
        oi_strat['gene_count'] = len(list(oi_chroms.keys())[0])
        return oi_strat
    
    def _make_arr_strat(self, arr_chroms):
        arr_strat = {'chromosomes': arr_chroms}
        arr_strat['strategy'] = {k: int(v, 2) for k, v in arr_chroms.items()}
        arr_strat['accuracy'] = {k: 0 for k in arr_chroms.keys()}
        arr_strat['gene_count'] = len(list(arr_chroms.keys())[0])
        return arr_strat
    
    def _make_bidask_strat(self, ba_chroms):
        ba_strat = {'chromosomes': ba_chroms}
        ba_strat['strategy'] = {k: int(v[1:], 2)*(1 if int(v[0]) else -1) for k, v in ba_chroms.items()}
        ba_strat['profitability'] = {k: 0 for k in ba_chroms.keys()}
        ba_strat['gene_count'] = len(list(ba_chroms.keys())[0])
        return ba_strat
    
    ''' New Strategy '''    
    def _make_oi_strat2(self, oi_chroms):
        oi_strat = {k: {'action': v, 'strategy': int(v[1:], 2)*(1 if int(v[0]) else -1), 'accuracy': 0} for k, v in oi_chroms.items()}
        return oi_strat, len(list(oi_chroms.keys())[0])
    
    def _make_arr_strat2(self, arr_chroms):
        arr_strat =  {k: {'action': v, 'strategy': int(v, 2), 'accuracy': 0} for k, v in arr_chroms.items()}
        return arr_strat, len(list(arr_chroms.keys())[0])
    
    def _make_bidask_strat2(self, ba_chroms):
        ba_strat = {k: {'action': v, 'strategy': int(v[1:], 2)*(1 if int(v[0]) else -1), 'profitability': 0} for k, v in ba_chroms.items()}
        return ba_strat, len(list(ba_chroms.keys())[0])
        
    
    ''' Old Matching '''
    def _match_oi_strat(self, market_state):
        temp_strength = []
        max_strength = 0
        max_accuracy = 0
        for cond in self._oi_strat['chromosomes'].keys():
            if all([(cond[x] == market_state[x] or cond[x] == '2') for x in range(self._oi_strat['gene_count'])]):
                strength = sum([cond[x] == market_state[x] for x in range(self._oi_strat['gene_count'])])
                if strength > max_strength:
                    temp_strength.clear()
                    temp_strength.append(cond)
                    max_strength = strength
                    max_accuracy = self._oi_strat['accuracy'][cond]
                elif strength == max_strength:
                    if self._oi_strat['accuracy'][cond] > max_accuracy:
                        temp_strength.clear()
                        max_accuracy = self._oi_strat['accuracy'][cond]
                    temp_strength.append(cond)          
        return {cond: self._oi_strat['accuracy'][cond] for cond in temp_strength}
    
    ''' New Matching '''
    def _match_oi_strat2(self, market_state):
        '''Returns all strategies with the maximum accuracy'''
        temp_strats = []
        max_strength = 0
        max_accuracy = 0
        for cond in self._oi_strat.keys():
            if all([(cond[x] == market_state[x] or cond[x] == '2') for x in range(self._oi_len)]):
                strength = sum([cond[x] == market_state[x] for x in range(self._oi_len)])
                if strength > max_strength:
                    temp_strats.clear()
                    temp_strats.append(cond)
                    max_strength = strength
                    max_accuracy = self._oi_strat[cond]['accuracy']
                elif strength == max_strength:
                    if self._oi_strat[cond]['accuracy'] > max_accuracy:
                        temp_strats.clear()
                        temp_strats.append(cond)
                        max_accuracy = self._oi_strat[cond]['accuracy']
                    elif self._oi_strat[cond]['accuracy'] == max_accuracy:
                        temp_strats.append(cond)         
        return temp_strats
    
    def _match_arr_strat2(self, market_state):
        '''Returns a randomly chosen strategy from all strategies with the maximum accuracy'''
        temp_strats = []
        max_strength = 0
        max_accuracy = 0
        for cond in self._arr_strat.keys():
            if all([(cond[x] == market_state[x] or cond[x] == '2') for x in range(self._arr_len)]):
                strength = sum([cond[x] == market_state[x] for x in range(self._arr_len)])
                if strength > max_strength:
                    temp_strats.clear()
                    temp_strats.append(cond)
                    max_strength = strength
                    max_accuracy = self._arr_strat[cond]['accuracy']
                elif strength == max_strength:
                    if self._arr_strat[cond]['accuracy'] > max_accuracy:
                        temp_strats.clear()
                        temp_strats.append(cond)
                        max_accuracy = self._arr_strat[cond]['accuracy']
                    elif self._arr_strat[cond]['accuracy'] == max_accuracy:
                        temp_strats.append(cond)         
        return temp_strats[random.randrange(len(temp_strats))]
                
    def _match_ask_strat(self, arrivals):
        '''Returns all strategies with the maximum accuracy'''
        temp_strats = []
        max_strength = 0
        max_accuracy = 0
        for cond in self._askadj_strat.keys():
            if all([(cond[x] == arrivals[x] or cond[x] == '2') for x in range(self._ask_len)]):
                strength = sum([cond[x] == arrivals[x] for x in range(self._ask_len)])
                if strength > max_strength:
                    temp_strats.clear()
                    temp_strats.append(cond)
                    max_strength = strength
                    max_accuracy = self._askadj_strat[cond]['accuracy']
                elif strength == max_strength:
                    if self._askadj_strat[cond]['accuracy'] > max_accuracy:
                        temp_strats.clear()
                        temp_strats.append(cond)
                        max_accuracy = self._askadj_strat[cond]['accuracy']
                    elif self._askadj_strat[cond]['accuracy'] == max_accuracy:
                        temp_strats.append(cond)         
        return temp_strats
    
    def _match_bid_strat(self, arrivals):
        '''Returns all strategies with the maximum accuracy'''
        temp_strats = []
        max_strength = 0
        max_accuracy = 0
        for cond in self._bidadj_strat.keys():
            if all([(cond[x] == arrivals[x] or cond[x] == '2') for x in range(self._bid_len)]):
                strength = sum([cond[x] == arrivals[x] for x in range(self._bid_len)])
                if strength > max_strength:
                    temp_strats.clear()
                    temp_strats.append(cond)
                    max_strength = strength
                    max_accuracy = self._bidadj_strat[cond]['accuracy']
                elif strength == max_strength:
                    if self._bidadj_strat[cond]['accuracy'] > max_accuracy:
                        temp_strats.clear()
                        temp_strats.append(cond)
                        max_accuracy = self._bidadj_strat[cond]['accuracy']
                    elif self._bidadj_strat[cond]['accuracy'] == max_accuracy:
                        temp_strats.append(cond)         
        return temp_strats
    
    ''' Make Orders '''                
    def _make_add_quote(self, time, side, price, quantity):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        return {'order_id': self._quote_sequence, 'trader_id': self.trader_id, 'timestamp': time, 
                'type': OType.ADD, 'quantity': quantity, 'side': side, 'price': price}
        
    def _make_cancel_quote(self, q, time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': q['trader_id'],
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}
    
    ''' Update Orderbook '''    
    def _update_midpoint(self, step, oib_signal):
        '''Compute change in inventory; obtain the most accurate oi strategies;
        average the forecast oi (if more than one best strategy); insert into midpoint update equation.'''
        delta_inv = self._position[step-1] - self._position[step-2]
        strategies = self._match_oi_strat2(oib_signal)
        flow = sum([self._oi_strat[c]['strategy'] for c in strategies])/len(strategies)
        self._mid += flow + int(self._c * delta_inv)
        
    def _make_spread(self, arr_signal, vol_signal):
        '''Obtain the most accurate arrival forecast; use as input to ask and bid strategies;
        average the most profitable adjustment strategies (if more than one); insert into
        ask and bid price adjustment; check for non-positive spread'''
        arrivals = self._match_arr_strat2(arr_signal)
        ask_strats = self._match_ask_strat(self._arr_strat[arrivals]['action'])
        bid_strats = self._match_bid_strat(self._arr_strat[arrivals]['action'])
        ask_adj = sum([self._askadj_strat[c]['strategy'] for c in ask_strats])/len(ask_strats)
        bid_adj = sum([self._bidadj_strat[c]['strategy'] for c in bid_strats])/len(bid_strats)
        ask = self._mid + min(self._a*vol_signal, self._b) + ask_adj
        bid = self._mid - min(self._a*vol_signal, self._b) + bid_adj
        while ask - bid <= 0:
            if random.random() > 0.5:
                ask+=1
            else:
                bid-=1
        return bid, ask
        
    def process_signal(self, step, signal):
        '''
        The signal is a dict with features of the market state: 
            arrival count: 16 bits
            order imbalance: 24 bits
            volatility: 10 period standard deviation of midpoint returns
            
        The midpoint forecast is a function of forecast order flow and inventory imbalance:
            mid(t) = mid(t-1) + D + c*I
            where D is the forecast order imbalance, I is the (change in) inventory imbalance and c is a parameter
            
        The market maker forecasts the new midpoint and sets the spread as a function of volatility:
            ask = mid + min(a*vol,b) +/- k
            bid = mid - min(a*vol,b) +/- k
            where a is sensitivity to volatility, b is a minimum and k is an adjustment based on arrival forecast
        '''
        # clear the collectors
        self.quote_collector.clear()
        self.cancel_collector.clear()
        
        # compute new midpoint
        self._update_midpoint(step, signal['oib'])
        
        # compute new spread
        bid, ask = self._make_spread(signal['arr'], signal['vol'])
        
        # cancel orders if inside new spread
        if bid < self._bid_book_prices[-1]:
            for p in range(bid+1, self._bid_book_prices[-1]+1):
                for q in self._bid_book[p]:
                    self.cancel_collector.append(self._make_cancel_quote(q, step))
                
            
            
            
        if ask > self._ask_book_prices[0]:
            for p in range(self._ask_book_prices[0], ask):
                for q in self._ask_book[p]:
                    self.cancel_collector.append(self._make_cancel_quote(q, step))
        
        
        # add new orders to make depth and/or establish new inside spread
        
        
        # update scores for predictors

        
        


class PennyJumper(ZITrader):
    '''
    PennyJumper jumps in front of best quotes when possible
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader), cancel_collector
    Public methods: confirm_trade_local (from ZITrader)
    '''
    trader_type = TType.PennyJumper
    
    def __init__(self, name, maxq, mpi):
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
    
    def _make_cancel_quote(self, q, time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': q['trader_id'],
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}

    def confirm_trade_local(self, confirm):
        '''PJ has at most one bid and one ask outstanding - if it executes, set price None'''
        if confirm['side'] == Side.BID:
            self._bid_quote = None
        else:
            self._ask_quote = None
            
    def process_signal(self, time, qsignal, q_taker):
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
                              

class Taker(ZITrader):
    '''
    Taker generates quotes (dicts) based on take probability.
        
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal 
    '''
    trader_type = TType.Taker

    def __init__(self, name, maxq):
        super().__init__(name, maxq)
        
    def process_signal(self, time, q_taker):
        '''Taker buys or sells with 50% probability.'''
        if random.random() < q_taker: # q_taker > 0.5 implies greater probability of a buy order
            return self._make_add_quote(time, Side.BID, 2000000)
        else:
            return self._make_add_quote(time, Side.ASK, 0)
        
        
class InformedTrader(ZITrader):
    '''
    InformedTrader generates quotes (dicts) based upon a fixed direction
    
    Subclass of ZITrader
    Public attributes: trader_type, quote_collector (from ZITrader)
    Public methods: process_signal
    '''
    trader_type = TType.Informed
    
    def __init__(self, name, maxq):
        ZITrader.__init__(self, name, maxq)
        self._side = random.choice([Side.BID, Side.ASK])
        self._price = 0 if self._side == Side.ASK else 2000000
        
    def process_signal(self, time):
        '''InformedTrader buys or sells pre-specified attribute.'''
        return self._make_add_quote(time, self._side, self._price)
