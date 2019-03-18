import bisect
import random

import pandas as pd

from mmabm.genetics import find_winners, make_strat, make_weights, match_strat_all, match_strat_random
from mmabm.genetics import new_genes_uf, new_genes_wf
from mmabm.shared import Side, OType, TType

            
class MarketMakerL:
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
    
    def __init__(self, name, maxq, arrInt, a, b, c, geneset, keep_pct, m, g_int):
        self.trader_id = name # trader id
        self._maxq = maxq
        self.arrInt = arrInt
        self._a = a
        self._b = b
        self._c = c
        self._bid_book = {}
        self._bid_book_prices = []
        self._ask_book = {}
        self._ask_book_prices = []
        self._quote_sequence = 0
        
        self.quote_collector = []
        self.cancel_collector = []
        
        self._bid = 0
        self._ask = 0
        self._mid = 0
        self._delta_inv = 0
        self._cash_flow = 0
        self.cash_flow_collector = []
        
        self._last_buy_prices = []
        self._last_sell_prices = []
        
        self._oi_strat, self._oi_len, self._oi_ngene = make_strat(geneset[0], 'accuracy')
        self._oi_keep = int(keep_pct * self._oi_ngene)
        self._oi_weights = make_weights(self._oi_keep)
        self._arr_strat, self._arr_len, self._arr_ngene = make_strat(geneset[1], 'accuracy', symm=False)
        self._arr_keep = int(keep_pct * self._arr_ngene)
        self._arr_weights = make_weights(self._arr_keep)
        self._spradj_strat, self._spr_len, self._spr_ngene = make_strat(geneset[2], 'rr_spread', maxi=False)
        self._spradj_keep = int(keep_pct * self._spr_ngene)
        self._spradj_weights = make_weights(self._spradj_keep)
        
        self._current_oi_strat = random.choice(list(self._oi_strat.keys()))
        self._current_arr_strat = random.choice(list(self._arr_strat.keys()))
        self._current_spradj_strat = []
        
        self._keep_p = keep_pct
        self._mutate_p = m
        self._genetic_int = g_int
        self.signal_collector = []


    ''' Update accuracy/rr_spread forecast '''                    
    def _update_oi_acc(self, actual):
        accuracy =  self._oi_strat[self._current_oi_strat]['accuracy']
        accuracy[0] += abs(actual - self._oi_strat[self._current_oi_strat]['strategy'])
        accuracy[1] += 1
        accuracy[-1] = 1000 - accuracy[0]/accuracy[1]
            
    def _update_arr_acc(self, actual):
        accuracy = self._arr_strat[self._current_arr_strat]['accuracy']
        accuracy[0] += abs(actual - self._arr_strat[self._current_arr_strat]['strategy'])
        accuracy[1] += 1
        accuracy[-1] = 1000 - accuracy[0]/accuracy[1]
        
    def _update_rspr(self, mid): # Using realized spread in ticks - maybe use relative realized spread?
        if self._last_sell_prices:
            for strat in self._current_spradj_strat:
                rr_spread = self._spradj_strat[strat]['rr_spread']
                rr_spread[0] += sum([x - mid for x in self._last_sell_prices])
                rr_spread[1] += len(self._last_sell_prices)
                rr_spread[-1] = rr_spread[0]/rr_spread[1]
        if self._last_buy_prices:
            for strat in self._current_spradj_strat:
                rr_spread = self._spradj_strat[strat]['rr_spread']
                rr_spread[0] += sum([mid - x for x in self._last_buy_prices])
                rr_spread[1] += len(self._last_buy_prices)
                rr_spread[-1] = rr_spread[0]/rr_spread[1]
                
    def _collect_signal(self, step, signal):
        keep = {k: v for k,v in signal.items()}
        keep['Step'] = step
        keep['OIStrat'] = self._current_oi_strat
        keep['OIStratAction'] = self._oi_strat[self._current_oi_strat]['strategy']
        keep['OIStratAccuracy'] = self._oi_strat[self._current_oi_strat]['accuracy'][-1]
        keep['ArrStrat'] = self._current_arr_strat
        keep['ArrStratAction'] = self._arr_strat[self._current_arr_strat]['strategy']
        keep['ArrStratAccuracy'] = self._arr_strat[self._current_arr_strat]['accuracy'][-1]
        for strat in self._current_spradj_strat:
            keep['SprStrat'] = strat
            keep['SprStratAction'] = self._spradj_strat[strat]['strategy']
            keep['SprStratAccuracy'] = self._spradj_strat[strat]['rr_spread'][-1]
        self.signal_collector.append(keep)
    
    ''' Handle Trades '''
    def confirm_trade_local(self, confirm):
        '''Modify _cash_flow and _delta_inv; update the local_book'''
        price = confirm['price']
        side = confirm['side']
        quantity = confirm['quantity']
        if side == Side.BID:
            self._last_buy_prices.append(price)
            self._cash_flow -= price*quantity/100000
            self._delta_inv += quantity
        else:
            self._last_sell_prices.append(price)
            self._cash_flow += price*quantity/100000
            self._delta_inv -= quantity
        self._modify_order(side, quantity, confirm['order_id'], price)
        
    def confirm_cross(self, confirm):
        ''' Could modify orderbook._confirm_trade to include incoming order_id, but:
        1. I want to avoid adding to the orderbook workload unnecessarily
        2. It is possible to obtain the order_id locally because it is guaranteed unique for a specific side/price
        During process_order, mm posts the order locally, then discovers (via a trade confirm) it has crossed the book.
        The trade must be recognized by adjusting cash flow and inventory 
        and then removing the order from the local book
        Eventually, the learning MM will not do this very frequently - if at all
        '''
        price = confirm['price']
        side = confirm['side']
        quantity = confirm['quantity']
        if side == Side.BID: # confirm side == BID means MM crossed (sold) with an ask order
            self._cash_flow += price*quantity
            self._delta_inv -= quantity
            mm_order = self._ask_book[price]['order_ids'][0]
            self._modify_order(Side.ASK, quantity, mm_order, price)
        else: # confirm side == ASK means MM  crossed (bought) with a buy order
            self._cash_flow -= price*quantity
            self._delta_inv += quantity
            mm_order = self._bid_book[price]['order_ids'][0]
            self._modify_order(Side.BID, quantity, mm_order, price)
        
    def cumulate_cashflow(self, step):
        self.cash_flow_collector.append({'mmid': self.trader_id, 'timestamp': step, 'cash_flow': self._cash_flow,
                                         'delta_inv': self._delta_inv})
    
    ''' Make Orders '''                
    def _make_add_quote(self, time, side, price, quantity):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        return {'order_id': self._quote_sequence, 'trader_id': self.trader_id, 'timestamp': time, 
                'type': OType.ADD, 'quantity': quantity, 'side': side, 'price': price}
        
    def _make_cancel_quote(self, q, time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': self.trader_id,
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
    def _update_midpoint(self, oib_signal, mid_signal):
        '''Compute change in inventory; obtain the most accurate oi strategies; insert into midpoint update equation.'''
        self._current_oi_strat = match_strat_random(oib_signal, 'accuracy', self._oi_strat, self._oi_len)
        self._mid = mid_signal + self._oi_strat[self._current_oi_strat]['strategy'] + int(self._c * self._delta_inv)
        
    def _make_spread(self, arr_signal, vol_signal):
        '''Obtain the most accurate arrival forecast; use as input to ask and bid strategies;
        average the most profitable adjustment strategies (if more than one); insert into
        ask and bid price adjustment; check for non-positive spread'''
        self._current_arr_strat = match_strat_random(arr_signal, 'accuracy', self._arr_strat, self._arr_len)
        self._current_spradj_strat = match_strat_all(self._arr_strat[self._current_arr_strat]['action'], 'rr_spread', self._spradj_strat, self._spr_len)
        spr_adj = sum([self._spradj_strat[c]['strategy'] for c in self._current_spradj_strat])/len(self._current_spradj_strat)
        self._ask = int(self._mid + round(max(self._a*vol_signal, self._b) + spr_adj/2))
        self._bid = int(self._mid - round(max(self._a*vol_signal, self._b) + spr_adj/2))
        while self._ask - self._bid <= 0:
            if random.random() > 0.5:
                self._ask += 1
            else:
                self._bid -= 1
    
    def _process_cancels(self, step):
        self.cancel_collector.clear()
        best_ask = self._ask_book_prices[0]
        if self._ask > best_ask:
            for p in range(best_ask, self._ask):
                if p in self._ask_book_prices:
                    self.cancel_collector.extend(self._make_cancel_quote(q, step) for q in self._ask_book[p]['orders'].values())
            for c in self.cancel_collector:
                self._remove_order(c['side'], c['price'], c['order_id'])
        best_bid = self._bid_book_prices[-1]
        if self._bid < best_bid:
            for p in range(self._bid + 1, best_bid + 1):
                if p in self._bid_book_prices:
                    self.cancel_collector.extend(self._make_cancel_quote(q, step) for q in self._bid_book[p]['orders'].values())
            for c in self.cancel_collector:
                self._remove_order(c['side'], c['price'], c['order_id'])
    
    def _update_ask_book(self, step, tob_bid):
        target_ask = max(self._ask, tob_bid + 1)
        if self._ask_book_prices:
            local_best_ask = self._ask_book_prices[0]
            if target_ask < local_best_ask:
                for p in range(target_ask, local_best_ask):
                    q = self._make_add_quote(step, Side.ASK, p, self._maxq)
                    self.quote_collector.append(q)
                    self._add_order(q)
                if self._ask_book[local_best_ask]['size'] < self._maxq:
                    q = self._make_add_quote(step, Side.ASK, local_best_ask, self._maxq - self._ask_book[local_best_ask]['size'])
                    self.quote_collector.append(q)
                    self._add_order(q)
            else:
                if self._ask_book[local_best_ask]['size'] < self._maxq:
                    q = self._make_add_quote(step, Side.ASK, local_best_ask, self._maxq - self._ask_book[local_best_ask]['size'])
                    self.quote_collector.append(q)
                    self._add_order(q)
        else:
            for p in range(target_ask, target_ask + 40):
                q = self._make_add_quote(step, Side.ASK, p, self._maxq)
                self.quote_collector.append(q)
                self._add_order(q)
        if len(self._ask_book_prices) < 20:
            for p in range(self._ask_book_prices[-1] + 1, self._ask_book_prices[-1] + 41 - len(self._ask_book_prices)):
                q = self._make_add_quote(step, Side.ASK, p, self._maxq)
                self.quote_collector.append(q)
                self._add_order(q)
        if len(self._ask_book_prices) > 60:
            for p in range(self._ask_book_prices[0] + 40, self._ask_book_prices[-1] + 1):
                self.cancel_collector.extend(self._make_cancel_quote(q, step) for q in self._ask_book[p]['orders'].values())
            for c in self.cancel_collector:
                self._remove_order(c['side'], c['price'], c['order_id'])
                
    def _update_bid_book(self, step, tob_ask):
        target_bid = min(self._bid, tob_ask - 1)
        if self._bid_book_prices:
            local_best_bid = self._bid_book_prices[-1]#  else target_bid - 40
            if target_bid > local_best_bid:
                for p in range(local_best_bid+1, target_bid+1):
                    q = self._make_add_quote(step, Side.BID, p, self._maxq)
                    self.quote_collector.append(q)
                    self._add_order(q)
                if self._bid_book[local_best_bid]['size'] < self._maxq:
                    q = self._make_add_quote(step, Side.BID, local_best_bid, self._maxq - self._bid_book[local_best_bid]['size'])
                    self.quote_collector.append(q)
                    self._add_order(q)
            else:
                if self._bid_book[local_best_bid]['size'] < self._maxq:
                    q = self._make_add_quote(step, Side.BID, local_best_bid, self._maxq - self._bid_book[local_best_bid]['size'])
                    self.quote_collector.append(q)
                    self._add_order(q)
        else:
            for p in range(target_bid - 39, target_bid + 1):
                q = self._make_add_quote(step, Side.BID, p, self._maxq)
                self.quote_collector.append(q)
                self._add_order(q)
        if len(self._bid_book_prices) < 20:
            for p in range(self._bid_book_prices[0] - 40 + len(self._bid_book_prices), self._bid_book_prices[0]):
                q = self._make_add_quote(step, Side.BID, p, self._maxq)
                self.quote_collector.append(q)
                self._add_order(q)
        if len(self._bid_book_prices) > 60:
            for p in range(self._bid_book_prices[0], self._bid_book_prices[-1] - 39):
                self.cancel_collector.extend(self._make_cancel_quote(q, step) for q in self._bid_book[p]['orders'].values())
            for c in self.cancel_collector:
                self._remove_order(c['side'], c['price'], c['order_id'])
                
    def seed_book(self, step, ask, bid):
        q = self._make_add_quote(step, Side.BID, bid, self._maxq)
        self.quote_collector.append(q)
        self._add_order(q)
        self._bid = bid
        q = self._make_add_quote(step, Side.ASK, ask, self._maxq)
        self.quote_collector.append(q)
        self._add_order(q)
        self._ask = ask
        self._mid = (ask+bid)/2
        
    def process_signal1(self, step, signal):
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
        self._update_rspr(signal['mid'])
        
        self._collect_signal(step, signal)
        
        #for strat in self._current_oi_strat:
            #print(step, strat, self._oi_strat[strat])
        
        # run genetics if it is time
        if not step % self._genetic_int:
            self._genetics_us()
            #self._genetics_ws()
        
        # compute new midpoint
        self._update_midpoint(signal['oib'], signal['mid'])
        
        # compute desired spread
        self._make_spread(signal['arr'], signal['vol'])
        
        # cancel old quotes
        self._process_cancels(step)
        
    def process_signal2(self, step, tob_bid, tob_ask):
        ''' Having cancelled unwanted orders in process_signal1, MML now adds orders to meet
        desired bid and ask, but will not cross the spread determined by other providers'''
        # clear the quote collector
        self.cancel_collector.clear()
        self.quote_collector.clear()
        
        # add new orders to make depth and/or establish new inside spread
        self._update_ask_book(step, tob_bid)
        self._update_bid_book(step, tob_ask)
        
        # update cash flow collector, reset inventory, clear recent prices
        self.cumulate_cashflow(step)
        self._delta_inv = 0
        self._last_buy_prices.clear()
        self._last_sell_prices.clear()
        
    ''' Genetic Algorithm Machinery '''
    def _get_winners(self):
        self._oi_strat = find_winners(self._oi_strat, self._oi_len, 'accuracy', self._oi_keep)
        self._arr_strat = find_winners(self._arr_strat, self._arr_len, 'accuracy', self._arr_keep)
        self._spradj_strat = find_winners(self._spradj_strat, self._spr_len, 'rr_spread', self._spradj_keep)

    def _genetics_us(self):
        self._get_winners()
        self._oi_strat = new_genes_uf(self._oi_strat, self._oi_ngene, self._oi_len, self._mutate_p, 5, 'accuracy', sym_mean)
        self._arr_strat = new_genes_uf(self._arr_strat, self._arr_ngene, self._arr_len, self._mutate_p, 5, 'accuracy', asym_mean)
        self._spradj_strat = new_genes_uf(self._spradj_strat, self._spr_ngene, self._spr_len, self._mutate_p, 3, 'rr_spread', sym_mean, maxi=False)
    
    def _genetics_ws(self):
        self._get_winners()
        self._oi_strat = new_genes_wf(self._oi_strat, self._oi_ngene, self._oi_weights, self._oi_len, self._mutate_p, 5, 'accuracy', sym_mean)
        self._arr_strat = new_genes_wf(self._arr_strat, self._arr_ngene, self._arr_weights, self._arr_len, self._mutate_p, 5, 'accuracy', asym_mean)
        self._spradj_strat = new_genes_wf(self._spradj_strat, self._spr_ngene, self._spradj_weights, self._spr_len, self._mutate_p, 3, 'rr_spread', sym_mean, maxi=False)
        
    def signal_collector_to_h5(self, filename):
        '''Append signal to an h5 file'''
        temp_df = pd.DataFrame(self.signal_collector)
        temp_df.to_hdf(filename, 'signal_%d' % self.trader_id, append=True, format='table', complevel=5, complib='blosc')

def sym_mean(s, a_len):
    a = '1' if s > 0 else '0'
    a += format(abs(s), 'b').rjust(a_len, '0')
    return a

def asym_mean(s, a_len):
    return format(s, 'b').rjust(a_len, '0')