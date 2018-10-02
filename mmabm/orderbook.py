import bisect
import pandas as pd

from mmabm.shared import Side, OType

class Orderbook(object):
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
        confirm_modify_collector and confirm_trade_collector are lists that carry information
        (dicts) from the order processor and/or matching engine to the traders
        trade_book is a list if trades in sequence
        _order_index identifies the sequence of orders in event time
        '''
        self.order_history = []
        self._bid_book = {}
        self._bid_book_prices = []
        self._ask_book = {}
        self._ask_book_prices = []
        self.confirm_trade_collector = []
        self._sip_collector = []
        self.trade_book = []
        self._order_index = 0
        self._ex_index = 0
        self._lookup = {}
        self.traded = False

    def add_order_to_history(self, order):
        '''Add an order (dict) to order_history'''
        self._order_index += 1
        self.order_history.append({'exid': self._order_index, 'order_id': order['order_id'], 'trader_id': order['trader_id'],
                                   'timestamp': order['timestamp'], 'type': order['type'].value, 'quantity': order['quantity'],
                                   'side': order['side'].value, 'price': order['price']})

    def add_order_to_book(self, order):
        '''
        Use insort to maintain on ordered list of prices which serve as pointers
        to the orders.
        '''
        book_order = {'order_id': order['order_id'], 'trader_id': order['trader_id'], 'timestamp': order['timestamp'],
                      'quantity': order['quantity'], 'side': order['side'], 'price': order['price']}
        self._ex_index += 1
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
            level['ex_ids'].append(self._ex_index)
            level['orders'][self._ex_index] = book_order
        else:
            bisect.insort(book_prices, order['price'])
            book[order['price']] = {'num_orders': 1, 'size': order['quantity'], 'ex_ids': [self._ex_index],
                                    'orders': {self._ex_index: book_order}}
        self._add_order_to_lookup(book_order['trader_id'], book_order['order_id'], self._ex_index)

    def _add_order_to_lookup(self, trader_id, order_id, ex_id):
        '''
        Add lookup for ex_id
        '''
        if trader_id in self._lookup.keys():
            self._lookup[trader_id][order_id] = ex_id
        else:
            self._lookup[trader_id] = {order_id: ex_id}

    def _remove_order(self, order_side, order_price, ex_id):
        '''Pop the order_id; if  order_id exists, updates the book.'''
        if order_side == Side.BID:
            book_prices = self._bid_book_prices
            book = self._bid_book
        else:
            book_prices = self._ask_book_prices
            book = self._ask_book
        is_order = book[order_price]['orders'].pop(ex_id, None)
        if is_order:
            level = book[order_price]
            level['num_orders'] -= 1
            level['size'] -= is_order['quantity']
            level['ex_ids'].remove(ex_id)
            if level['num_orders'] == 0:
                book_prices.remove(order_price)
            del self._lookup[is_order['trader_id']][is_order['order_id']]

    def _modify_order(self, order_side, order_quantity, ex_id, order_price):
        '''Modify order quantity; if quantity is 0, removes the order.'''
        book = self._bid_book if order_side == Side.BID else self._ask_book
        if order_quantity < book[order_price]['orders'][ex_id]['quantity']:
            book[order_price]['size'] -= order_quantity
            book[order_price]['orders'][ex_id]['quantity'] -= order_quantity
        else:
            self._remove_order(order_side, order_price, ex_id)

    def _add_trade_to_book(self, resting_trader_id, resting_order_id, resting_timestamp,
                           incoming_trader_id, incoming_order_id, timestamp, price, quantity, side):
        '''Add trades (dicts) to the trade_book list.'''
        self.trade_book.append({'resting_trader_id': resting_trader_id, 'resting_order_id': resting_order_id, 'resting_timestamp': resting_timestamp,
                                'incoming_trader_id': incoming_trader_id, 'incoming_order_id': incoming_order_id, 'timestamp': timestamp, 'price': price,
                                'quantity': quantity, 'side': side.value})

    def _confirm_trade(self, timestamp, order_side, order_quantity, order_id, order_price, trader_id):
        '''Add trade confirmation to confirm_trade_collector list.'''
        self.confirm_trade_collector.append({'timestamp': timestamp, 'trader': trader_id, 'order_id': order_id,
                                             'quantity': order_quantity, 'side': order_side, 'price': order_price})

    def process_order(self, order):
        '''Check for a trade (match); if so call _match_trade, otherwise modify book(s).'''
        self.traded = False
        self.add_order_to_history(order)
        if order['type'] == OType.ADD:
            if order['side'] == Side.BID:
                if order['price'] >= self._ask_book_prices[0]:
                    self._match_trade(order)
                else:
                    self.add_order_to_book(order)
            else: #order['side'] == 'sell'
                if order['price'] <= self._bid_book_prices[-1]:
                    self._match_trade(order)
                else:
                    self.add_order_to_book(order)
        else:
            ex_id = self._lookup[order['trader_id']][order['order_id']]
            if order['type'] == OType.CANCEL:
                self._remove_order(order['side'], order['price'], ex_id)
            else: #order['type'] == 'modify'
                self._modify_order(order['side'], order['quantity'], ex_id, order['price'])

    def _match_trade(self, order):
        '''Match orders to generate trades, update books.'''
        self.traded = True
        self.confirm_trade_collector.clear()
        if order['side'] == Side.BID:
            book_prices = self._ask_book_prices
            book = self._ask_book
            remainder = order['quantity']
            while remainder > 0:
                if book_prices:
                    price = book_prices[0]
                    if order['price'] >= price:
                        ex_id = book[price]['ex_ids'][0]
                        book_order = book[price]['orders'][ex_id]
                        if remainder >= book_order['quantity']:
                            self._confirm_trade(order['timestamp'], book_order['side'], book_order['quantity'], book_order['order_id'],
                                                book_order['price'], book_order['trader_id'])
                            self._add_trade_to_book(book_order['trader_id'], book_order['order_id'], book_order['timestamp'],
                                                    order['trader_id'], order['order_id'], order['timestamp'],
                                                    book_order['price'], book_order['quantity'], order['side'])
                            self._remove_order(book_order['side'], book_order['price'], ex_id)
                            remainder -= book_order['quantity']
                        else:
                            self._confirm_trade(order['timestamp'], book_order['side'], remainder, book_order['order_id'],
                                                book_order['price'], book_order['trader_id'])
                            self._add_trade_to_book(book_order['trader_id'], book_order['order_id'], book_order['timestamp'],
                                                    order['trader_id'], order['order_id'], order['timestamp'],
                                                    book_order['price'], remainder, order['side'])
                            self._modify_order(book_order['side'], remainder, ex_id, book_order['price'])
                            break
                    else:
                        order['quantity'] = remainder
                        self.add_order_to_book(order)
                        break
                else:
                    print('Ask Market Collapse with order {0}'.format(order))
                    break
        else:
            book_prices = self._bid_book_prices
            book = self._bid_book
            remainder = order['quantity']
            while remainder > 0:
                if book_prices:
                    price = book_prices[-1]
                    if order['price'] <= price:
                        ex_id = book[price]['ex_ids'][0]
                        book_order = book[price]['orders'][ex_id]
                        if remainder >= book_order['quantity']:
                            self._confirm_trade(order['timestamp'], book_order['side'], book_order['quantity'], book_order['order_id'],
                                                book_order['price'], book_order['trader_id'])
                            self._add_trade_to_book(book_order['trader_id'], book_order['order_id'], book_order['timestamp'],
                                                    order['trader_id'], order['order_id'], order['timestamp'],
                                                    book_order['price'], book_order['quantity'], order['side'])
                            self._remove_order(book_order['side'], book_order['price'], ex_id)
                            remainder -= book_order['quantity']
                        else:
                            self._confirm_trade(order['timestamp'], book_order['side'], remainder, book_order['order_id'],
                                                book_order['price'], book_order['trader_id'])
                            self._add_trade_to_book(book_order['trader_id'], book_order['order_id'], book_order['timestamp'],
                                                    order['trader_id'], order['order_id'], order['timestamp'],
                                                    book_order['price'], remainder, order['side'])
                            self._modify_order(book_order['side'], remainder, ex_id, book_order['price'])
                            break
                    else:
                        order['quantity'] = remainder
                        self.add_order_to_book(order)
                        break
                else:
                    print('Bid Market Collapse with order {0}'.format(order))
                    break

    def order_history_to_h5(self, filename):
        '''Append order history to an h5 file, clear the order_history'''
        temp_df = pd.DataFrame(self.order_history)
        temp_df.to_hdf(filename, 'orders', append=True, format='table', complevel=5, complib='blosc')
        self.order_history.clear()

    def trade_book_to_h5(self, filename):
        '''Append trade_book to an h5 file, clear the trade_book'''
        temp_df = pd.DataFrame(self.trade_book)
        temp_df.to_hdf(filename, 'trades', append=True, format='table', complevel=5, complib='blosc')
        self.trade_book.clear()

    def sip_to_h5(self, filename):
        '''Append _sip_collector to an h5 file, clear the _sip_collector'''
        temp_df = pd.DataFrame(self._sip_collector)
        temp_df.to_hdf(filename, 'tob', append=True, format='table', complevel=5, complib='blosc')
        self._sip_collector.clear()

    def report_top_of_book(self, now_time):
        '''Update the top-of-book prices and sizes'''
        best_bid_price = self._bid_book_prices[-1]
        best_bid_size = self._bid_book[best_bid_price]['size']
        best_ask_price = self._ask_book_prices[0]
        best_ask_size = self._ask_book[best_ask_price]['size']
        tob = {'timestamp': now_time, 'best_bid': best_bid_price, 'best_ask': best_ask_price, 'bid_size': best_bid_size, 'ask_size': best_ask_size}
        self._sip_collector.append(tob)
        return tob
    