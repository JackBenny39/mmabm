import bisect

from mmabm.shared import Side


class Localbook:

    def __init__(self):
        self.bid_book = {}
        self.bid_book_prices = []
        self.ask_book = {}
        self.ask_book_prices = []

    # Orderbook Bookkeeping with List
    def add_order(self, order):
        '''
        Use insort to maintain on ordered list of prices which serve as pointers
        to the orders.
        '''
        book_order = {'order_id': order['order_id'], 'timestamp': order['timestamp'], 'quantity': order['quantity'],
                      'side': order['side'], 'price': order['price']}
        if order['side'] == Side.BID:
            book_prices = self.bid_book_prices
            book = self.bid_book
        else:
            book_prices = self.ask_book_prices
            book = self.ask_book
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

    def remove_order(self, order_side, order_price, order_id):
        '''Pop the order_id; if  order_id exists, updates the book.'''
        if order_side == Side.BID:
            book_prices = self.bid_book_prices
            book = self.bid_book
        else:
            book_prices = self.ask_book_prices
            book = self.ask_book
        is_order = book[order_price]['orders'].pop(order_id, None)
        if is_order:
            level = book[order_price]
            level['num_orders'] -= 1
            level['size'] -= is_order['quantity']
            level['order_ids'].remove(order_id)
            if level['num_orders'] == 0:
                book_prices.remove(order_price)

    def modify_order(self, order_side, order_quantity, order_id, order_price):
        '''Modify order quantity; if quantity is 0, removes the order.'''
        book = self.bid_book if order_side == Side.BID else self.ask_book
        if order_quantity < book[order_price]['orders'][order_id]['quantity']:
            book[order_price]['size'] -= order_quantity
            book[order_price]['orders'][order_id]['quantity'] -= order_quantity
        else:
            self.remove_order(order_side, order_price, order_id)