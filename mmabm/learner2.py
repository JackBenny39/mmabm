import random


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

        self._bid = 0
        self._ask = 0
        self._mid = 0
        self._delta_inv = 0
        self._cash_flow = 0
        self.cash_flow_collector = []

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
        self._mid = (ask + bid) / 2

    # Handle Trades
    def confirm_trade_local(self, confirm):
        '''Modify _cash_flow and _delta_inv; update the local_book'''
        price = confirm['price']
        side = confirm['side']
        quantity = confirm['quantity']
        if side == Side.BID:
            #self._last_buy_prices.append(price)
            self._cash_flow -= price*quantity/100000
            self._delta_inv += quantity
        else:
            #self._last_sell_prices.append(price)
            self._cash_flow += price*quantity/100000
            self._delta_inv -= quantity
        self._localbook.modify_order(side, quantity, confirm['order_id'], price)
        
    def cumulate_cashflow(self, step):
        self.cash_flow_collector.append({'mmid': self.trader_id, 'timestamp': step, 'cash_flow': self._cash_flow,
                                         'delta_inv': self._delta_inv})

    # Make Orders
    def _make_add_quote(self, time, side, price, quantity):
        '''Make one add quote (dict)'''
        self._quote_sequence += 1
        return {'order_id': self._quote_sequence, 'trader_id': self.trader_id, 'timestamp': time, 
                'type': OType.ADD, 'quantity': quantity, 'side': side, 'price': price}
        
    def _make_cancel_quote(self, q, time):
        return {'type': OType.CANCEL, 'timestamp': time, 'order_id': q['order_id'], 'trader_id': self.trader_id,
                'quantity': q['quantity'], 'side': q['side'], 'price': q['price']}

    # Update Orderbook
    def _update_midpoint(self, bid, ask):
        self._mid = (bid + ask) / 2

    def _make_spread(self, bid, ask):
        self._ask = ask + random.randint(-2, 2)
        self._bid = bid + random.randint(-2, 2)
        while self._ask - self._bid <= 0:
            if random.random() > 0.5:
                self._ask += 1
            else:
                self._bid -= 1

    def _process_cancels(self, step):
        self.cancel_collector.clear()
        best_ask = self._localbook.ask_book_prices[0]
        if self._ask > best_ask:
            for p in range(best_ask, self._ask):
                if p in self._localbook.ask_book_prices:
                    self.cancel_collector.extend(self._make_cancel_quote(q, step) for q in self._localbook.ask_book[p]['orders'].values())
            for c in self.cancel_collector:
                self._localbook.remove_order(c['side'], c['price'], c['order_id'])
        best_bid = self._localbook.bid_book_prices[-1]
        if self._bid < best_bid:
            for p in range(self._bid + 1, best_bid + 1):
                if p in self._localbook.bid_book_prices:
                    self.cancel_collector.extend(self._make_cancel_quote(q, step) for q in self._localbook.bid_book[p]['orders'].values())
            for c in self.cancel_collector:
                self._localbook.remove_order(c['side'], c['price'], c['order_id'])

    def _update_ask_book(self, step, tob_bid):
        target_ask = max(self._ask, tob_bid + 1)
        if self._localbook.ask_book_prices:
            local_best_ask = self._localbook.ask_book_prices[0]
            if target_ask < local_best_ask:
                for p in range(target_ask, local_best_ask):
                    q = self._make_add_quote(step, Side.ASK, p, self._maxq)
                    self.quote_collector.append(q)
                    self._localbook.add_order(q)
                if self._localbook.ask_book[local_best_ask]['size'] < self._maxq:
                    q = self._make_add_quote(step, Side.ASK, local_best_ask, self._maxq - self._localbook.ask_book[local_best_ask]['size'])
                    self.quote_collector.append(q)
                    self._localbook.add_order(q)
            else:
                if self._localbook.ask_book[local_best_ask]['size'] < self._maxq:
                    q = self._make_add_quote(step, Side.ASK, local_best_ask, self._maxq - self._localbook.ask_book[local_best_ask]['size'])
                    self.quote_collector.append(q)
                    self._localbook.add_order(q)
        else:
            for p in range(target_ask, target_ask + 40):
                q = self._make_add_quote(step, Side.ASK, p, self._maxq)
                self.quote_collector.append(q)
                self._localbook.add_order(q)
        if len(self._localbook.ask_book_prices) < 20:
            for p in range(self._localbook.ask_book_prices[-1] + 1, self._localbook.ask_book_prices[-1] + 41 - len(self._localbook.ask_book_prices)):
                q = self._make_add_quote(step, Side.ASK, p, self._maxq)
                self.quote_collector.append(q)
                self._localbook.add_order(q)
        if len(self._localbook.ask_book_prices) > 60:
            for p in range(self._localbook.ask_book_prices[0] + 40, self._localbook.ask_book_prices[-1] + 1):
                self.cancel_collector.extend(self._make_cancel_quote(q, step) for q in self._localbook.ask_book[p]['orders'].values())
            for c in self.cancel_collector:
                self._localbook.remove_order(c['side'], c['price'], c['order_id'])
                
    def _update_bid_book(self, step, tob_ask):
        target_bid = min(self._bid, tob_ask - 1)
        if self._localbook.bid_book_prices:
            local_best_bid = self._localbook.bid_book_prices[-1]#  else target_bid - 40
            if target_bid > local_best_bid:
                for p in range(local_best_bid + 1, target_bid + 1):
                    q = self._make_add_quote(step, Side.BID, p, self._maxq)
                    self.quote_collector.append(q)
                    self._localbook.add_order(q)
                if self._localbook.bid_book[local_best_bid]['size'] < self._maxq:
                    q = self._make_add_quote(step, Side.BID, local_best_bid, self._maxq - self._localbook.bid_book[local_best_bid]['size'])
                    self.quote_collector.append(q)
                    self._localbook.add_order(q)
            else:
                if self._localbook.bid_book[local_best_bid]['size'] < self._maxq:
                    q = self._make_add_quote(step, Side.BID, local_best_bid, self._maxq - self._localbook.bid_book[local_best_bid]['size'])
                    self.quote_collector.append(q)
                    self._localbook.add_order(q)
        else:
            for p in range(target_bid - 39, target_bid + 1):
                q = self._make_add_quote(step, Side.BID, p, self._maxq)
                self.quote_collector.append(q)
                self._localbook.add_order(q)
        if len(self._localbook.bid_book_prices) < 20:
            for p in range(self._localbook.bid_book_prices[0] - 40 + len(self._localbook.bid_book_prices), self._localbook.bid_book_prices[0]):
                q = self._make_add_quote(step, Side.BID, p, self._maxq)
                self.quote_collector.append(q)
                self._localbook.add_order(q)
        if len(self._localbook.bid_book_prices) > 60:
            for p in range(self._localbook.bid_book_prices[0], self._localbook.bid_book_prices[-1] - 39):
                self.cancel_collector.extend(self._make_cancel_quote(q, step) for q in self._localbook.bid_book[p]['orders'].values())
            for c in self.cancel_collector:
                self._localbook.remove_order(c['side'], c['price'], c['order_id'])

    # Process Signal
    def process_signal1(self, step, signal):
        '''
        The signal is a tuple with features of the market state: 
            order imbalance: 24 bits
            
        The midpoint is a function of forecast order flow and inventory imbalance:
            mid(t) = mid(t-1) + D + c*I
            where D is the forecast order imbalance, I is the (change in) inventory imbalance and c is a parameter
        '''

        # Update predictor accuracy
        self._oi.update_accuracies(signal[0]) # actual oi is signal[0]

        # Run genetics if it is time
        if not step % self._genetic_int:
            self._oi.new_genes_uf()
            #self._oi.new_genes_wf()

        # Compute new midpoint
        self._update_midpoint(signal[1], signal[2]) # signal[1] is the bid; signal[2] is the ask
        
        # Compute desired spread
        self._make_spread(signal[1], signal[2]) # signal[1] is the bid; signal[2] is the ask

        # Cancel old quotes
        self._process_cancels(step)

    def process_signal2(self, step, tob_bid, tob_ask):
        ''' Having cancelled unwanted orders in process_signal1, MML now adds orders to meet
        desired bid and ask, but will not cross the spread determined by other providers'''

        # Clear the collectors
        self.cancel_collector.clear()
        self.quote_collector.clear()
        
        # Add new orders to make depth and/or establish new inside spread (Start here 20190410)
        self._update_ask_book(step, tob_bid)
        self._update_bid_book(step, tob_ask)
        
        # update cash flow collector, reset inventory, clear recent prices
        self.cumulate_cashflow(step)
        self._delta_inv = 0
        #self._last_buy_prices.clear()
        #self._last_sell_prices.clear()