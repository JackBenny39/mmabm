import bisect
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
    Spread: function of volatility and expected order flow
    Depth: predetermined minimum and maximum range (for now)
    Price range: emergent outcome
    
    Public attributes:
    Public methods:
    Private attributes:
    Private methods:
    '''
    trader_type = TType.MarketMaker
    
    def __init__(self, name, maxq, a, b, c, geneset):
        self.trader_id = name # trader id
        self._maxq = maxq
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
        
        self._current_oi_strat = []
        self._current_arr_strat = None
        self._current_ask_strat = []
        self._current_bid_strat = []


    ''' New Strategy '''    
    def _make_oi_strat2(self, oi_chroms):
        oi_strat = {k: {'action': v, 'strategy': int(v[1:], 2)*(1 if int(v[0]) else -1), 'accuracy': [0, 0, 0]} for k, v in oi_chroms.items()}
        return oi_strat, len(list(oi_chroms.keys())[0])
    
    def _make_arr_strat2(self, arr_chroms):
        arr_strat =  {k: {'action': v, 'strategy': int(v, 2), 'accuracy': [0, 0, 0]} for k, v in arr_chroms.items()}
        return arr_strat, len(list(arr_chroms.keys())[0])
    
    def _make_bidask_strat2(self, ba_chroms):
        ba_strat = {k: {'action': v, 'strategy': int(v[1:], 2)*(1 if int(v[0]) else -1), 'profitability': [0, 0, 0]} for k, v in ba_chroms.items()}
        return ba_strat, len(list(ba_chroms.keys())[0])

    ''' New Matching '''
    def _match_oi_strat2(self, market_state):
        '''Returns all strategies with the maximum accuracy'''
        self._current_oi_strat.clear()
        max_strength = 0
        max_accuracy = 0
        for cond in self._oi_strat.keys():
            if all([(cond[x] == market_state[x] or cond[x] == '2') for x in range(self._oi_len)]):
                strength = sum([cond[x] == market_state[x] for x in range(self._oi_len)])
                if strength > max_strength:
                    self._current_oi_strat.clear()
                    self._current_oi_strat.append(cond)
                    max_strength = strength
                    max_accuracy = self._oi_strat[cond]['accuracy'][0]
                elif strength == max_strength:
                    if self._oi_strat[cond]['accuracy'][0] > max_accuracy:
                        self._current_oi_strat.clear()
                        self._current_oi_strat.append(cond)
                        max_accuracy = self._oi_strat[cond]['accuracy'][0]
                    elif self._oi_strat[cond]['accuracy'][0] == max_accuracy:
                        self._current_oi_strat.append(cond)
    
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
                    max_accuracy = self._arr_strat[cond]['accuracy'][0]
                elif strength == max_strength:
                    if self._arr_strat[cond]['accuracy'][0] > max_accuracy:
                        temp_strats.clear()
                        temp_strats.append(cond)
                        max_accuracy = self._arr_strat[cond]['accuracy'][0]
                    elif self._arr_strat[cond]['accuracy'][0] == max_accuracy:
                        temp_strats.append(cond)         
        self._current_arr_strat = temp_strats[random.randrange(len(temp_strats))]
                
    def _match_ask_strat(self, arrivals):
        '''Returns all strategies with the maximum accuracy'''
        self._current_ask_strat.clear()
        max_strength = 0
        max_profits = 0
        for cond in self._askadj_strat.keys():
            if all([(cond[x] == arrivals[x] or cond[x] == '2') for x in range(self._ask_len)]):
                strength = sum([cond[x] == arrivals[x] for x in range(self._ask_len)])
                if strength > max_strength:
                    self._current_ask_strat.clear()
                    self._current_ask_strat.append(cond)
                    max_strength = strength
                    max_profits = self._askadj_strat[cond]['profitability'][0]
                elif strength == max_strength:
                    if self._askadj_strat[cond]['profitability'][0] > max_profits:
                        self._current_ask_strat.clear()
                        self._current_ask_strat.append(cond)
                        max_profits = self._askadj_strat[cond]['profitability'][0]
                    elif self._askadj_strat[cond]['profitability'][0] == max_profits:
                        self._current_ask_strat.append(cond)         
    
    def _match_bid_strat(self, arrivals):
        '''Returns all strategies with the maximum accuracy'''
        self._current_bid_strat.clear()
        max_strength = 0
        max_profits = 0
        for cond in self._bidadj_strat.keys():
            if all([(cond[x] == arrivals[x] or cond[x] == '2') for x in range(self._bid_len)]):
                strength = sum([cond[x] == arrivals[x] for x in range(self._bid_len)])
                if strength > max_strength:
                    self._current_bid_strat.clear()
                    self._current_bid_strat.append(cond)
                    max_strength = strength
                    max_profits = self._bidadj_strat[cond]['profitability'][0]
                elif strength == max_strength:
                    if self._bidadj_strat[cond]['profitability'][0] > max_profits:
                        self._current_bid_strat.clear()
                        self._current_bid_strat.append(cond)
                        max_profits = self._bidadj_strat[cond]['profitability'][0]
                    elif self._bidadj_strat[cond]['profitability'][0] == max_profits:
                        self._current_bid_strat.append(cond)

    ''' Update accuracy/profitability forecast '''                    
    def _update_oi_acc(self, actual):
        for strat in self._current_oi_strat: #sub out references
            self._oi_strat[strat]['accuracy'][0] += abs(actual - self._oi_strat[strat]['strategy'])
            self._oi_strat[strat]['accuracy'][1] += 1
            self._oi_strat[strat]['accuracy'][-1] = self._oi_strat[strat]['accuracy'][0]/self._oi_strat[strat]['accuracy'][1]
            
    def _update_arr_acc(self, actual): #sub out references
        self._arr_strat[self._current_arr_strat]['accuracy'][0] += abs(actual - self._arr_strat[self._current_arr_strat]['strategy'])
        self._arr_strat[self._current_arr_strat]['accuracy'][1] += 1
        self._arr_strat[self._current_arr_strat]['accuracy'][-1] = self._arr_strat[self._current_arr_strat]['accuracy'][0]/self._oi_strat[self._current_arr_strat]['accuracy'][1]
        
    def _update_ask_pft(self):
        pass
    
    def _update_bid_pft(self):
        pass
    
    
    ''' Make Orders '''                
    def _make_add_quote(self, time, side, price, quantity):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        return {'order_id': self._quote_sequence, 'trader_id': self.trader_id, 'timestamp': time, 
                'type': OType.ADD, 'quantity': quantity, 'side': side, 'price': price}
        
    def _make_cancel_quote(self, q, time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': q['trader_id'],
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}
        
    ''' Orderbook Bookkeeping with List'''
    def _add_order(self, order):
        '''
        Use insort to maintain on ordered list of prices which serve as pointers
        to the orders.
        '''
        book_order = {'order_id': order['order_id'], 'timestamp': order['timestamp'], 'quantity': order['quantity'],
                      'side': order['side'], 'price': order['price']}
        if order['side'] == Side.BID:
            book_prices = self._bid_book_prices
            book = self._bid_book
        else:
            book_prices = self._ask_book_prices
            book = self._ask_book
        if order['price'] in book_prices:
            level = book[order['price']]
            level['num_orders'] += 1
            level['size'] += order['quantity']
            level['order_ids'].append(book_order['order_id'])
            level['orders'][book_order['order_id']] = book_order
        else:
            bisect.insort(book_prices, order['price'])
            book[order['price']] = {'num_orders': 1, 'size': order['quantity'], 'order_ids': [book_order['order_id']],
                                    'orders': {book_order['order_id']: book_order}}

    def _remove_order(self, order_side, order_price, order_id):
        '''Pop the order_id; if  order_id exists, updates the book.'''
        if order_side == Side.BID:
            book_prices = self._bid_book_prices
            book = self._bid_book
        else:
            book_prices = self._ask_book_prices
            book = self._ask_book
        is_order = book[order_price]['orders'].pop(order_id, None)
        if is_order:
            level = book[order_price]
            level['num_orders'] -= 1
            level['size'] -= is_order['quantity']
            level['order_ids'].remove(order_id)
            if level['num_orders'] == 0:
                book_prices.remove(order_price)

    def _modify_order(self, order_side, order_quantity, order_id, order_price):
        '''Modify order quantity; if quantity is 0, removes the order.'''
        book = self._bid_book if order_side == Side.BID else self._ask_book
        if order_quantity < book[order_price]['orders'][order_id]['quantity']:
            book[order_price]['size'] -= order_quantity
            book[order_price]['orders'][order_id]['quantity'] -= order_quantity
        else:
            self._remove_order(order_side, order_price, order_id)

    ''' Update Orderbook '''    
    def _update_midpoint(self, step, oib_signal):
        '''Compute change in inventory; obtain the most accurate oi strategies;
        average the forecast oi (if more than one best strategy); insert into midpoint update equation.'''
        delta_inv = self._position[step-1] - self._position[step-2]
        self._match_oi_strat2(oib_signal)
        flow = sum([self._oi_strat[c]['strategy'] for c in self._current_oi_strat])/len(self._current_oi_strat)
        self._mid += flow + int(self._c * delta_inv)
        
    def _make_spread(self, arr_signal, vol_signal):
        '''Obtain the most accurate arrival forecast; use as input to ask and bid strategies;
        average the most profitable adjustment strategies (if more than one); insert into
        ask and bid price adjustment; check for non-positive spread'''
        self._match_arr_strat2(arr_signal)
        self._match_ask_strat(self._arr_strat[self._current_arr_strat]['action'])
        self._match_bid_strat(self._arr_strat[self._current_arr_strat]['action'])
        ask_adj = sum([self._askadj_strat[c]['strategy'] for c in self._current_ask_strat])/len(self._current_ask_strat)
        bid_adj = sum([self._bidadj_strat[c]['strategy'] for c in self._current_bid_strat])/len(self._current_bid_strat)
        ask = self._mid + min(self._a*vol_signal, self._b) + ask_adj
        bid = self._mid - min(self._a*vol_signal, self._b) + bid_adj
        while ask - bid <= 0:
            if random.random() > 0.5:
                ask+=1
            else:
                bid-=1
        return bid, ask
    
    def _update_ask_book(self, step, ask):
        best_ask = self._ask_book_prices[0]
        if ask < best_ask:
            for p in range(ask, best_ask):
                q = self._make_add_quote(step, Side.ASK, p, self._maxq)
                self.quote_collector.append(q)
                self._add_order(q)
            if self._ask_book[best_ask]['size'] < self._maxq:
                q = self._make_add_quote(step, Side.ASK, best_ask, self._maxq - self._ask_book[best_ask]['size'])
                self.quote_collector.append(q)
                self._add_order(q)
        elif ask > best_ask:
            for p in range(best_ask, ask):
                for q in self._ask_book[p]:
                    self.cancel_collector.append(self._make_cancel_quote(q, step))
                for c in self.cancel_collector:
                    self._remove_order(c['side'], c['price'], c['order_id'])
        else:
            if self._ask_book[best_ask]['size'] < self._maxq:
                q = self._make_add_quote(step, Side.ASK, best_ask, self._maxq - self._ask_book[best_ask]['size'])
                self.quote_collector.append(q)
                self._add_order(q)
        if len(self._ask_book_prices) < 20:
            for p in range(self._ask_book_prices[-1] + 1, self._ask_book_prices[-1] + 41 - len(self._ask_book_prices)):
                q = self._make_add_quote(step, Side.ASK, p, self._maxq)
                self.quote_collector.append(q)
                self._add_order(q)
        if len(self._ask_book_prices) > 60:
            for p in range(self._ask_book_prices[-1] - 20, self._ask_book_prices[-1]+1):
                for q in self._ask_book[p]:
                    self.cancel_collector.append(self._make_cancel_quote(q, step))
                for c in self.cancel_collector:
                    self._remove_order(c['side'], c['price'], c['order_id'])
                
    def _update_bid_book(self, step, bid):
        best_bid = self._bid_book_prices[-1]
        if bid > best_bid:
            for p in range(best_bid+1, bid+1):
                q = self._make_add_quote(step, Side.BID, p, self._maxq)
                self.quote_collector.append(q)
                self._add_order(q)
            if self._bid_book[best_bid]['size'] < self._maxq:
                q = self._make_add_quote(step, Side.BID, best_bid, self._maxq - self._bid_book[best_bid]['size'])
                self.quote_collector.append(q)
                self._add_order(q)
        elif bid < best_bid:
            for p in range(bid+1, best_bid+1):
                for q in self._bid_book[p]:
                    self.cancel_collector.append(self._make_cancel_quote(q, step))
                for c in self.cancel_collector:
                    self._remove_order(c['side'], c['price'], c['order_id'])
        else:
            if self._bid_book[best_bid]['size'] < self._maxq:
                q = self._make_add_quote(step, Side.BID, best_bid, self._maxq - self._bid_book[best_bid]['size'])
                self.quote_collector.append(q)
                self._add_order(q)
        if len(self._bid_book_prices) < 20:
            for p in range(self._bid_book_prices[0] - 40 + self._bid_book_prices, self._bid_book_prices[0]):
                q = self._make_add_quote(step, Side.BID, p, self._maxq)
                self.quote_collector.append(q)
                self._add_order(q)
        if len(self._bid_book_prices) > 60:
            for p in range(self._bid_book_prices[0], self._bid_book_prices[0]+21):
                for q in self._bid_book[p]:
                    self.cancel_collector.append(self._make_cancel_quote(q, step))
                for c in self.cancel_collector:
                    self._remove_order(c['side'], c['price'], c['order_id'])
        
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
        # update scores for predictors
        self._update_oi_acc(signal['oibv'])
        self._update_arr_acc(signal['arrv'])
        self._update_ask_pft()
        self._update_bid_pft()
        
        
        # clear the collectors
        self.quote_collector.clear()
        self.cancel_collector.clear()
        
        # compute new midpoint
        self._update_midpoint(step, signal['oib'])
        
        # compute new spread
        bid, ask = self._make_spread(signal['arr'], signal['vol'])
        
        # cancel old orders or add new orders to make depth and/or establish new inside spread
        self._update_ask_book(step, ask)
        self._update_bid_book(step, bid)
        

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
