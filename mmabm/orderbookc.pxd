from mmabm.sharedc cimport Side


cdef class Orderbook:
    cdef list _bid_book_prices, _ask_book_prices, _sip_collector
    cdef public list order_history, confirm_modify_collector, confirm_trade_collector, trade_book
    cdef dict _bid_book, _ask_book, _lookup
    cdef unsigned int _order_index, _ex_index
    cdef public bint traded
    
    cpdef add_order_to_history(self, dict order)
    cpdef add_order_to_book(self, dict order)
    cdef void _add_order_to_lookup(self, unsigned int trader_id, unsigned int order_id, unsigned int ex_id)
    cdef void _remove_order(self, Side order_side, unsigned int order_price, unsigned int ex_id)
    cdef void _modify_order(self, Side order_side, unsigned int order_quantity, unsigned int ex_id, unsigned int order_price)
    cdef void _add_trade_to_book(self, unsigned int resting_trader_id, unsigned int resting_order_id, unsigned int resting_timestamp,
                                 unsigned int incoming_trader_id, unsigned int incoming_order_id,
                                 unsigned int timestamp, unsigned int price, unsigned int quantity, Side side)
    cdef void _confirm_trade(self, unsigned int timestamp, Side order_side, unsigned int order_quantity, unsigned int order_id,
                             unsigned int order_price, unsigned int trader_id)
    cdef void _confirm_modify(self, unsigned int timestamp, Side order_side, unsigned int order_quantity, unsigned int order_id,
                              unsigned int trader_id)
    cpdef process_order(self, dict order)
    cdef void _match_trade(self, dict order)
    cpdef report_top_of_book(self, unsigned int now_time)