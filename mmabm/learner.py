import bisect
import random

import numpy as np

from mmabm.shared import Side, OType, TType

            
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
        
        self._oi_strat, self._oi_len, self._oi_ngene = self._make_oi_strat2(geneset[0])
        self._oi_keep = int(keep_pct * self._oi_ngene)
        self._oi_weights = self._make_weights(self._oi_keep)
        self._arr_strat, self._arr_len, self._arr_ngene = self._make_arr_strat2(geneset[1])
        self._arr_keep = int(keep_pct * self._arr_ngene)
        self._arr_weights = self._make_weights(self._arr_keep)
        self._spradj_strat, self._spr_len, self._spr_ngene = self._make_spread_strat2(geneset[2])
        self._spradj_keep = int(keep_pct * self._spr_ngene)
        self._spradj_weights = self._make_weights(self._spradj_keep)
        
        self._current_oi_strat = []
        self._current_arr_strat = random.choice(list(self._arr_strat.keys()))
        self._current_spradj_strat = []
        
        self._keep_p = keep_pct
        self._mutate_p = m
        self._genetic_int = g_int


    ''' New Strategy '''    
    def _make_oi_strat2(self, oi_chroms):
        oi_strat = {k: {'action': v, 'strategy': int(v[1:], 2)*(1 if int(v[0]) else -1), 'accuracy': [0, 0, 1000]} for k, v in oi_chroms.items()}
        return oi_strat, len(list(oi_chroms.keys())[0]), len(oi_strat)
    
    def _make_arr_strat2(self, arr_chroms):
        arr_strat =  {k: {'action': v, 'strategy': int(v, 2), 'accuracy': [0, 0, 1000]} for k, v in arr_chroms.items()}
        return arr_strat, len(list(arr_chroms.keys())[0]), len(arr_strat)
    
    def _make_spread_strat2(self, ba_chroms):
        ba_strat = {k: {'action': v, 'strategy': int(v[1:], 2)*(1 if int(v[0]) else -1), 'rr_spread': [0, 0, 0]} for k, v in ba_chroms.items()}
        return ba_strat, len(list(ba_chroms.keys())[0]), len(ba_strat)
    
    def _make_weights(self, l):
        denom = sum([j for j in range(1, l+1)])
        numer = reversed([j for j in range(1, l+1)])
        return np.cumsum([k/denom for k in numer])

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
                    max_accuracy = self._oi_strat[cond]['accuracy'][-1]
                elif strength == max_strength:
                    if self._oi_strat[cond]['accuracy'][-1] > max_accuracy:
                        self._current_oi_strat.clear()
                        self._current_oi_strat.append(cond)
                        max_accuracy = self._oi_strat[cond]['accuracy'][-1]
                    elif self._oi_strat[cond]['accuracy'][-1] == max_accuracy:
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
                    max_accuracy = self._arr_strat[cond]['accuracy'][-1]
                elif strength == max_strength:
                    if self._arr_strat[cond]['accuracy'][-1] > max_accuracy:
                        temp_strats.clear()
                        temp_strats.append(cond)
                        max_accuracy = self._arr_strat[cond]['accuracy'][-1]
                    elif self._arr_strat[cond]['accuracy'][-1] == max_accuracy:
                        temp_strats.append(cond)         
        self._current_arr_strat = random.choice(temp_strats)
                
    def _match_spread_strat(self, arrivals):
        '''Returns all strategies with the maximum accuracy'''
        self._current_spradj_strat.clear()
        max_strength = 0
        max_rs = 0
        for cond in self._spradj_strat.keys():
            if all([(cond[x] == arrivals[x] or cond[x] == '2') for x in range(self._spr_len)]):
                strength = sum([cond[x] == arrivals[x] for x in range(self._spr_len)])
                if strength > max_strength:
                    self._current_spradj_strat.clear()
                    self._current_spradj_strat.append(cond)
                    max_strength = strength
                    max_rs = self._spradj_strat[cond]['rr_spread'][-1]
                elif strength == max_strength:
                    if self._spradj_strat[cond]['rr_spread'][-1] > max_rs:
                        self._current_spradj_strat.clear()
                        self._current_spradj_strat.append(cond)
                        max_rs = self._spradj_strat[cond]['rr_spread'][-1]
                    elif self._spradj_strat[cond]['rr_spread'][-1] == max_rs:
                        self._current_spradj_strat.append(cond)

    ''' Update accuracy/rr_spread forecast '''                    
    def _update_oi_acc(self, actual):
        for strat in self._current_oi_strat:
            accuracy =  self._oi_strat[strat]['accuracy']
            accuracy[0] += abs(actual - self._oi_strat[strat]['strategy'])
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
        '''Compute change in inventory; obtain the most accurate oi strategies;
        average the forecast oi (if more than one best strategy); insert into midpoint update equation.'''
        self._match_oi_strat2(oib_signal)
        flow = sum([self._oi_strat[c]['strategy'] for c in self._current_oi_strat])/len(self._current_oi_strat)
        self._mid = mid_signal + flow + int(self._c * self._delta_inv)
        
    def _make_spread(self, arr_signal, vol_signal):
        '''Obtain the most accurate arrival forecast; use as input to ask and bid strategies;
        average the most profitable adjustment strategies (if more than one); insert into
        ask and bid price adjustment; check for non-positive spread'''
        self._match_arr_strat2(arr_signal)
        self._match_spread_strat(self._arr_strat[self._current_arr_strat]['action'])
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
    def _find_winners(self):
        oi_allk = '2' * self._oi_len
        oi_all = {oi_allk: self._oi_strat.pop(oi_allk)}
        self._oi_strat = dict(sorted(self._oi_strat.items(), key=lambda kv: kv[1]['accuracy'][2], reverse=True)[:self._oi_keep - 1])
        self._oi_strat.update(oi_all)
        
        arr_allk = '2' * self._arr_len
        arr_all = {arr_allk: self._arr_strat.pop(arr_allk)}
        self._arr_strat = dict(sorted(self._arr_strat.items(), key=lambda kv: kv[1]['accuracy'][2], reverse=True)[:self._arr_keep - 1])
        self._arr_strat.update(arr_all)
        
        spr_allk = '2' * self._spr_len
        spr_all = {spr_allk: self._spradj_strat.pop(spr_allk)}
        self._spradj_strat = dict(sorted(self._spradj_strat.items(), key=lambda kv: kv[1]['rr_spread'][2], reverse=True)[:self._spradj_keep - 1])
        self._spradj_strat.update(spr_all)
        
    def _uniform_selection(self):
        oi_parents = list(self._oi_strat.keys())
        arr_parents = list(self._arr_strat.keys())
        spr_parents = list(self._spradj_strat.keys())
        o1, o2 = tuple(random.sample(oi_parents, 2))
        a1, a2 = tuple(random.sample(arr_parents, 2))
        s1, s2 = tuple(random.sample(spr_parents, 2))
        print(o1, o2)
        print(a1, a2)
        print(s1, s2)
    
    def _weighted_selection(self):
        oi_parents = list(self._oi_strat.keys())
        arr_parents = list(self._arr_strat.keys())
        spr_parents = list(self._spradj_strat.keys())
        o1, o2 = tuple(random.choices(oi_parents, cum_weights=self._oi_weights, k=2))
        a1, a2 = tuple(random.choices(arr_parents, cum_weights=self._arr_weights, k=2))
        s1, s2 = tuple(random.choices(spr_parents, cum_weights=self._spradj_weights, k=2))
        print(o1, o2)
        print(a1, a2)
        print(s1, s2)
        
    def _crossover(self, p1, p2, x):
        return p1[:x] + p2[x:]
    
    def _mutate_cond(self, c, l):
        c[random.randrange(l)] = random.randrange(3)
        return c
 
    def _mutate_action(self, c, l):
        c[random.randrange(l)] = random.randrange(2)
        return c
    
    def _oi_genes_us(self):
        # Step 1: get the genes
        oi_parents = list(self._oi_strat.keys())
        # Step 2: update the strategy dict with unique new children
        while len(self._oi_strat) < self._oi_ngene:
            # Choose two parents - uniform selection
            o1, o2 = tuple(random.sample(oi_parents, 2))
            # Random uniform crossover
            x = random.randrange(self._oi_len)
            o = o1[:x] + o2[x:]
            # Random uniform mutate
            if random.random() < self._mutate_p:
                z = random.randrange(self._oi_len)
                o = o[:z] + str(random.randrange(3)) + o[z+1:]
            # Check if new child differs from current parents
            if o not in list(self._oi_strat.keys()):
                # Update child action & strategy
                y = random.random()
                if y < 0.333: # choose parent 1
                    action = self._oi_strat[o1]['action']
                    strategy = self._oi_strat[o1]['strategy']
                elif y > 0.667: # choose parent 2
                    action = self._oi_strat[o2]['action']
                    strategy = self._oi_strat[o2]['strategy']
                else: # average parent 1 & 2
                    strategy = int((self._oi_strat[o1]['strategy'] + self._oi_strat[o2]['strategy']) / 2)
                    action = '1' if strategy > 0 else '0'
                    action += format(abs(strategy), 'b').rjust(5, '0')
                # Update accuracy - weighted average
                a0 = self._oi_strat[o1]['accuracy'][0] + self._oi_strat[o2]['accuracy'][0]
                a1 = self._oi_strat[o1]['accuracy'][1] + self._oi_strat[o2]['accuracy'][1]
                a2 = a0/a1 if a1 > 0 else 0
                accuracy = [a0, a1, 1000 - a2]
                # Add new child to strategy dict
                self._oi_strat.update({o: {'action': action, 'strategy': strategy, 'accuracy': accuracy}})
            
    def _arr_genes_us(self):
        # Step 1: get the genes
        arr_parents = list(self._arr_strat.keys())
        # Step 2: update the strategy dict with unique new children
        while len(self._arr_strat) < self._arr_ngene:
            # Choose two parents - uniform selection
            r1, r2 = tuple(random.sample(arr_parents, 2))
            # Random uniform crossover
            x = random.randrange(self._arr_len)
            r = r1[:x] + r2[x:]
            # Random uniform mutate
            if random.random() < self._mutate_p:
                z = random.randrange(self._arr_len)
                r = r[:z] + str(random.randrange(3)) + r[z+1:]
            # Check if new child differs from current parents
            if r not in list(self._arr_strat.keys()):
                # Update child action & strategy
                y = random.random()
                if y < 0.333: # choose parent 1
                    action = self._arr_strat[r1]['action']
                    strategy = self._arr_strat[r1]['strategy']
                elif y > 0.667: # choose parent 2
                    action = self._arr_strat[r2]['action']
                    strategy = self._arr_strat[r2]['strategy']
                else: # average parent 1 & 2
                    strategy = int((self._arr_strat[r1]['strategy'] + self._arr_strat[r2]['strategy']) / 2)
                    action = format(strategy, 'b').rjust(5, '0')
                # Update accuracy - weighted average
                a0 = self._arr_strat[r1]['accuracy'][0] + self._arr_strat[r2]['accuracy'][0]
                a1 = self._arr_strat[r1]['accuracy'][1] + self._arr_strat[r2]['accuracy'][1]
                a2 = a0/a1 if a1 > 0 else 0
                accuracy = [a0, a1, 1000 - a2]
                # Add new child to strategy dict
                self._arr_strat.update({r: {'action': action, 'strategy': strategy, 'accuracy': accuracy}})
            
    def _spr_genes_us(self):
        # Step 1: get the genes
        spr_parents = list(self._spradj_strat.keys())
        # Step 2: update the strategy dict with unique new children
        while len(self._spradj_strat) < self._spr_ngene:
            # Choose two parents - uniform selection
            s1, s2 = tuple(random.sample(spr_parents, 2))
            # Random uniform crossover
            x = random.randrange(self._spr_len)
            s = s1[:x] + s2[x:]
            # Random uniform mutate
            if random.random() < self._mutate_p:
                z = random.randrange(self._spr_len)
                s = s[:z] + str(random.randrange(3)) + s[z+1:]
            # Check if new child differs from current parents
            if s not in list(self._spradj_strat.keys()):
                # Update child action & strategy
                y = random.random()
                if y < 0.333: # choose parent 1
                    action = self._spradj_strat[s1]['action']
                    strategy = self._spradj_strat[s1]['strategy']
                elif y > 0.667: # choose parent 2
                    action = self._spradj_strat[s2]['action']
                    strategy = self._spradj_strat[s2]['strategy']
                else: # average parent 1 & 2
                    strategy = int((self._spradj_strat[s1]['strategy'] + self._spradj_strat[s2]['strategy']) / 2)
                    action = '1' if strategy > 0 else '0'
                    action += format(abs(strategy), 'b').rjust(3, '0')
                # Update accuracy - weighted average
                a0 = self._spradj_strat[s1]['rr_spread'][0] + self._spradj_strat[s2]['rr_spread'][0]
                a1 = self._spradj_strat[s1]['rr_spread'][1] + self._spradj_strat[s2]['rr_spread'][1]
                a2 = a0/a1 if a1 > 0 else 0
                rr_spread = [a0, a1, a2]
                # Add new child to strategy dict
                self._spradj_strat.update({s: {'action': action, 'strategy': strategy, 'rr_spread': rr_spread}})
    
    def _genetics_us(self):
        self._find_winners()
        self._oi_genes_us()
        self._arr_genes_us()
        self._spr_genes_us()
        
    def _oi_genes_ws(self):
        # Step 1: get the genes
        oi_parents = list(self._oi_strat.keys())
        # Step 2: update the strategy dict with unique new children
        while len(self._oi_strat) < self._oi_ngene:
            # Choose two parents - weighted selection
            o1, o2 = tuple(random.choices(oi_parents, cum_weights=self._oi_weights, k=2))
            # Random uniform crossover
            x = random.randrange(self._oi_len)
            o = o1[:x] + o2[x:]
            # Random uniform mutate
            if random.random() < self._mutate_p:
                z = random.randrange(self._oi_len)
                o = o[:z] + str(random.randrange(3)) + o[z+1:]
            # Check if new child differs from current parents
            if o not in list(self._oi_strat.keys()):
                # Update child action & strategy
                y = random.random()
                if y < 0.333: # choose parent 1
                    action = self._oi_strat[o1]['action']
                    strategy = self._oi_strat[o1]['strategy']
                elif y > 0.667: # choose parent 2
                    action = self._oi_strat[o2]['action']
                    strategy = self._oi_strat[o2]['strategy']
                else: # average parent 1 & 2
                    strategy = int((self._oi_strat[o1]['strategy'] + self._oi_strat[o2]['strategy']) / 2)
                    action = '1' if strategy > 0 else '0'
                    action += format(abs(strategy), 'b').rjust(5, '0')
                # Update accuracy - weighted average
                a0 = self._oi_strat[o1]['accuracy'][0] + self._oi_strat[o2]['accuracy'][0]
                a1 = self._oi_strat[o1]['accuracy'][1] + self._oi_strat[o2]['accuracy'][1]
                a2 = a0/a1 if a1 > 0 else 0
                accuracy = [a0, a1, 1000 - a2]
                # Add new child to strategy dict
                self._oi_strat.update({o: {'action': action, 'strategy': strategy, 'accuracy': accuracy}})
            
    def _arr_genes_ws(self):
        # Step 1: get the genes
        arr_parents = list(self._arr_strat.keys())
        # Step 2: update the strategy dict with unique new children
        while len(self._arr_strat) < self._arr_ngene:
            # Choose two parents - weighted selection
            r1, r2 = tuple(random.choices(arr_parents, cum_weights=self._arr_weights, k=2))
            # Random uniform crossover
            x = random.randrange(self._arr_len)
            r = r1[:x] + r2[x:]
            # Random uniform mutate
            if random.random() < self._mutate_p:
                z = random.randrange(self._arr_len)
                r = r[:z] + str(random.randrange(3)) + r[z+1:]
            # Check if new child differs from current parents
            if r not in list(self._arr_strat.keys()):
                # Update child action & strategy
                y = random.random()
                if y < 0.333: # choose parent 1
                    action = self._arr_strat[r1]['action']
                    strategy = self._arr_strat[r1]['strategy']
                elif y > 0.667: # choose parent 2
                    action = self._arr_strat[r2]['action']
                    strategy = self._arr_strat[r2]['strategy']
                else: # average parent 1 & 2
                    strategy = int((self._arr_strat[r1]['strategy'] + self._arr_strat[r2]['strategy']) / 2)
                    action = format(strategy, 'b').rjust(5, '0')
                # Update accuracy - weighted average
                a0 = self._arr_strat[r1]['accuracy'][0] + self._arr_strat[r2]['accuracy'][0]
                a1 = self._arr_strat[r1]['accuracy'][1] + self._arr_strat[r2]['accuracy'][1]
                a2 = a0/a1 if a1 > 0 else 0
                accuracy = [a0, a1, 1000 - a2]
                # Add new child to strategy dict
                self._arr_strat.update({r: {'action': action, 'strategy': strategy, 'accuracy': accuracy}})
            
    def _spr_genes_ws(self):
        # Step 1: get the genes
        spr_parents = list(self._spradj_strat.keys())
        # Step 2: update the strategy dict with unique new children
        while len(self._spradj_strat) < self._spr_ngene:
            # Choose two parents - weighted selection
            s1, s2 = tuple(random.choices(spr_parents, cum_weights=self._spradj_weights, k=2))
            # Random uniform crossover
            x = random.randrange(self._spr_len)
            s = s1[:x] + s2[x:]
            # Random uniform mutate
            if random.random() < self._mutate_p:
                z = random.randrange(self._spr_len)
                s = s[:z] + str(random.randrange(3)) + s[z+1:]
            # Check if new child differs from current parents
            if s not in list(self._spradj_strat.keys()):
                # Update child action & strategy
                y = random.random()
                if y < 0.333: # choose parent 1
                    action = self._spradj_strat[s1]['action']
                    strategy = self._spradj_strat[s1]['strategy']
                elif y > 0.667: # choose parent 2
                    action = self._spradj_strat[s2]['action']
                    strategy = self._spradj_strat[s2]['strategy']
                else: # average parent 1 & 2
                    strategy = int((self._spradj_strat[s1]['strategy'] + self._spradj_strat[s2]['strategy']) / 2)
                    action = '1' if strategy > 0 else '0'
                    action += format(abs(strategy), 'b').rjust(3, '0')
                # Update accuracy - weighted average
                a0 = self._spradj_strat[s1]['rr_spread'][0] + self._spradj_strat[s2]['rr_spread'][0]
                a1 = self._spradj_strat[s1]['rr_spread'][1] + self._spradj_strat[s2]['rr_spread'][1]
                a2 = a0/a1 if a1 > 0 else 0
                rr_spread = [a0, a1, a2]
                # Add new child to strategy dict
                self._spradj_strat.update({s: {'action': action, 'strategy': strategy, 'rr_spread': rr_spread}})
    
    def _genetics_ws(self):
        self._find_winners()
        self._oi_genes_ws()
        self._arr_genes_ws()
        self._spr_genes_ws()
