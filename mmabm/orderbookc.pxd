from mmabm.sharedc cimport Side


cdef class Orderbook:
    cdef list _bid_book_prices, _ask_book_prices, _sip_collector
    cdef public list order_history, confirm_trade_collector, trade_book
    cdef dict _bid_book, _ask_book, _lookup
    cdef int _order_index, _ex_index
    cdef public bint traded
    
    cpdef add_order_to_history(self, dict order)
    cpdef add_order_to_book(self, dict order)
    cdef void _add_order_to_lookup(self, int trader_id, int order_id, int ex_id)
    cdef void _remove_order(self, Side order_side, int order_price, int ex_id)
    cdef void _modify_order(self, Side order_side, int order_quantity, int ex_id, int order_price)
    cdef void _add_trade_to_book(self, int resting_trader_id, int resting_order_id, int resting_timestamp,
                                 int incoming_trader_id, int incoming_order_id,
                                 int timestamp, int price, int quantity, Side side)
    cdef void _confirm_trade(self, int timestamp, Side order_side, int order_quantity, int order_id,
                             int order_price, int trader_id)
    cpdef process_order(self, dict order)
    cdef void _match_trade(self, dict order)
    cpdef dict report_top_of_book(self, int now_time)