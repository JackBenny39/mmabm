# distutils: language = c++

from libcpp.list cimport list as clist
from libcpp.map cimport map
from libcpp.pair cimport pair

import pandas as pd

from cython.operator cimport postincrement as inc, postdecrement as dec, dereference as deref

from mmabm.sharedc cimport Side, OType


cdef class Orderbook:
    '''
    Orderbook tracks, processes and matches orders.
    
    Orderbook is a set of linked lists and dictionaries containing trades, bids and asks.
    One dictionary contains a history of all orders;
    two other dictionaries contain priced bid and ask orders with linked lists for access;
    one dictionary contains trades matched with orders on the book.
    Orderbook also provides methods for storing and retrieving orders and maintaining a 
    history of the book.
    Public attributes: order_history, confirm_modify_collector, confirm_trade_collector,
    trade_book and traded.
    Public methods: add_order_to_book(), process_order(), order_history_to_h5(), trade_book_to_h5(),
    sip_to_h5() and report_top_of_book()
    '''
    
    def __init__(self):
        '''
        Initialize the Orderbook with a set of empty lists and dicts and other defaults
        
        order_history is a list of all incoming orders (dicts) in the order received
        _bid_book_prices and _ask_book_prices are linked (sorted) lists of bid and ask prices
        which serve as pointers to:
        _bid_book and _ask_book: dicts of current order book state and dicts of orders
        the oder id lists maintain time priority for each order at a given price.
        confirm_modify_collector and confirm_trade_collector are lists that carry information (dicts) from the
        order processor and/or matching engine to the traders
        trade_book is a list if trades in sequence
        _order_index identifies the sequence of orders in event time
        '''
        self.order_history = []
        self._bids = BookSide()
        self._asks = BookSide()
        self.confirm_trade_collector = []
        self._sip_collector = []
        self.trade_book = []
        self._order_index = 0
        self._ex_index = 0
        self._lookup = LookUp()
        self.traded = False
       
    cpdef add_order_to_history(self, dict order):
        self._order_index += 1
        self.order_history.append({'exid': self._order_index, 'order_id': order['order_id'], 'trader_id': order['trader_id'], 
                                   'timestamp': order['timestamp'], 'type': order['type'], 'quantity': order['quantity'], 
                                   'side': order['side'], 'price': order['price']})
    
    cpdef add_order_to_book(self, int trader_id, int order_id, int timestamp, int quantity, Side side, int price):
        cdef BookSide *b = &self._bids if side == Side.BID else &self._asks
        l = deref(b).find(price)
        if l == deref(b).end():
            l = deref(b).insert(OneLevel(price, Level(0, 0, Quotes()))).first
        inc(deref(l).second.cnt)
        deref(l).second.qty = deref(l).second.qty + quantity
        q = deref(l).second.quotes.insert(deref(l).second.quotes.end(), Quote(trader_id, order_id, timestamp, quantity, side, price))
        self._lookup.insert(OneLookUp(OrderId(trader_id, order_id), BLQ(b, l, q)))
       
    cdef void _remove_order(self, int trader_id, int order_id, int quantity):
        cdef OrderId oid = OrderId(trader_id, order_id)
        cdef BLQ *blq = &self._lookup[oid]
        b = blq.bs_ptr
        l = blq.bs_it
        q = blq.q_it
        deref(l).second.qty = deref(l).second.qty - quantity
        deref(l).second.quotes.erase(q)
        dec(deref(l).second.cnt)
        if not deref(l).second.cnt:
            b.erase(l)
        self._lookup.erase(oid)
           
    cdef void _modify_order(self, int trader_id, int order_id, int quantity):
        cdef OrderId oid = OrderId(trader_id, order_id)
        cdef BLQ *blq = &self._lookup[oid]
        b = blq.bs_ptr
        l = blq.bs_it
        q = blq.q_it
        deref(l).second.qty = deref(l).second.qty - quantity
        deref(q).qty = deref(q).qty - quantity
        if not deref(q).qty:
            deref(l).second.quotes.erase(q)
            dec(deref(l).second.cnt)
            if not deref(l).second.cnt:
                b.erase(l)
            self._lookup.erase(oid)
            
    cdef void _add_trade_to_book(self, int resting_trader_id, int resting_order_id, int resting_timestamp,
                                 int incoming_trader_id, int incoming_order_id,
                                 int timestamp, int price, int quantity, Side side):
        self.trade_book.append({'resting_trader_id': resting_trader_id, 'resting_order_id': resting_order_id, 'resting_timestamp': resting_timestamp, 
                                'incoming_trader_id': incoming_trader_id, 'incoming_order_id': incoming_order_id, 'timestamp': timestamp, 'price': price,
                                'quantity': quantity, 'side': side})

    cdef void _confirm_trade(self, int timestamp, Side order_side, int order_quantity, int order_id,
                             int order_price, int trader_id):
        self.confirm_trade_collector.append({'timestamp': timestamp, 'trader': trader_id, 'order_id': order_id, 
                                             'quantity': order_quantity, 'side': order_side, 'price': order_price})
        
    cdef BookTop get_bid(self):
        if self._bids.empty():
            return BookTop(0, 0)
        else: 
            return BookTop(deref(self._bids.rbegin()).first, deref(self._bids.rbegin()).second.qty)
    
    cdef BookTop get_ask(self):
        if self._asks.empty():
            return BookTop(0, 0)
        else: 
            return BookTop(deref(self._asks.begin()).first, deref(self._asks.begin()).second.qty)
            
    cpdef process_order(self, dict order):
        self.traded = False
        #self.add_order_to_history(order)
        
        if order['type'] == OType.ADD:
            if order['side'] == Side.BID:
                if order['price'] >= self.get_ask().first:
                    self._match_trade(order['trader_id'], order['order_id'], order['timestamp'],  
                                      order['quantity'], order['side'], order['price'])
                else:
                    self.add_order_to_book(order['trader_id'], order['order_id'], order['timestamp'],  
                                           order['quantity'], order['side'], order['price'])
            else:
                if order['price'] <= self.get_bid().first:
                    self._match_trade(order['trader_id'], order['order_id'], order['timestamp'],  
                                      order['quantity'], order['side'], order['price'])
                else:
                    self.add_order_to_book(order['trader_id'], order['order_id'], order['timestamp'],  
                                           order['quantity'], order['side'], order['price'])
        else:
            if order['type'] == OType.CANCEL:
                self._remove_order(order['trader_id'], order['order_id'], order['quantity'])
            else:
                self._modify_order(order['trader_id'], order['order_id'], order['quantity'])
                        
    cdef void _match_trade(self, int trader_id, int order_id, int timestamp, int quantity, Side side, int price):
        self.traded = True
        self.confirm_trade_collector.clear()
        
        cdef int best
        
        if side == Side.BID:
            while quantity > 0:
                best = self.get_ask().first
                if best:
                    if price >= best:
                        qq = self._asks[best].quotes.front()
                        if quantity >= qq.qty:
                            self._confirm_trade(timestamp, qq.side, qq.qty, qq.order_id, 
                                                qq.price, qq.trader_id)
                            self._add_trade_to_book(qq.trader_id, qq.order_id, qq.timestamp,
                                                    trader_id, order_id, timestamp,
                                                    qq.price, qq.qty, side)
                            quantity -= qq.qty
                            self._remove_order(qq.trader_id, qq.order_id, qq.qty)
                        else:
                            self._confirm_trade(timestamp, qq.side, quantity, qq.order_id, 
                                                qq.price, qq.trader_id)
                            self._add_trade_to_book(qq.trader_id, qq.order_id, qq.timestamp,
                                                    trader_id, order_id, timestamp,
                                                    qq.price, quantity, side)
                            self._modify_order(qq.trader_id, qq.order_id, qq.qty)
                            break
                    else:
                        self.add_order_to_book(trader_id, order_id, timestamp, quantity, side, price)
                        break
                else:
                    print('Ask Market Collapse with order {0} - {1}'.format(trader_id, order_id))
                    break
        else:
            while quantity > 0:
                best = self.get_bid().first
                if best:
                    if price <= best:
                        qq = self._bids[best].quotes.front()
                        if quantity >= qq.qty:
                            self._confirm_trade(timestamp, qq.side, qq.qty, qq.order_id, 
                                                qq.price, qq.trader_id)
                            self._add_trade_to_book(qq.trader_id, qq.order_id, qq.timestamp,
                                                    trader_id, order_id, timestamp,
                                                    qq.price, qq.qty, side)
                            quantity -= qq.qty
                            self._remove_order(qq.trader_id, qq.order_id, qq.qty)
                        else:
                            self._confirm_trade(timestamp, qq.side, quantity, qq.order_id, 
                                                qq.price, qq.trader_id)
                            self._add_trade_to_book(qq.trader_id, qq.order_id, qq.timestamp,
                                                    trader_id, order_id, timestamp,
                                                    qq.price, quantity, side)
                            self._modify_order(qq.trader_id, qq.order_id, qq.qty)
                            break
                    else:
                        self.add_order_to_book(trader_id, order_id, timestamp, quantity, side, price)
                        break
                else:
                    print('Bid Market Collapse with order {0} - {1}'.format(trader_id, order_id))
                    break
    
    def order_history_to_h5(self, filename):
        temp_df = pd.DataFrame(self.order_history)
        temp_df.to_hdf(filename, 'orders', append=True, format='table', complevel=5, complib='blosc') 
        self.order_history.clear()
        
    def trade_book_to_h5(self, filename):
        temp_df = pd.DataFrame(self.trade_book)
        temp_df.to_hdf(filename, 'trades', append=True, format='table', complevel=5, complib='blosc') 
        self.trade_book.clear()
        
    def sip_to_h5(self, filename):
        temp_df = pd.DataFrame(self._sip_collector)
        temp_df.to_hdf(filename, 'tob', append=True, format='table', complevel=5, complib='blosc')
        self._sip_collector.clear()
    
    cpdef dict report_top_of_book(self, int now_time):
        best_ask = self.get_ask()
        best_bid = self.get_bid()
        cdef dict tob = {'timestamp': now_time, 'best_bid': best_bid.first, 'best_ask': best_ask.first, 'bid_size': best_bid.second, 'ask_size': best_ask.second}
        self._sip_collector.append(tob)
        return tob
        