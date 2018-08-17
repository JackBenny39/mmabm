# distutils: language = c++

from libcpp.list cimport list as clist
from libcpp.map cimport map
from libcpp.pair cimport pair

from mmabm.sharedc cimport OType, Side

ctypedef struct Quote:
    int trader_id
    int order_id
    int timestamp
    int qty
    Side side
    int price
    
ctypedef Quote* q_ptr
    
ctypedef clist[Quote] Quotes
    
ctypedef struct Level:
    int cnt
    int qty
    Quotes quotes

ctypedef map[int, Level] BookSide
ctypedef pair[int, Level] OneLevel

ctypedef struct BLQ:
    BookSide* bs_ptr
    map[int, Level].iterator bs_it
    clist[Quote].iterator q_it
    
ctypedef pair[int, int] OrderId
ctypedef map[OrderId, BLQ] LookUp
ctypedef pair[OrderId, BLQ] OneLookUp

ctypedef pair[int, int] BookTop


cdef class Orderbook:
    cdef list _sip_collector
    cdef BookSide _bids, _asks
    cdef public list order_history, confirm_trade_collector, trade_book
    cdef LookUp _lookup
    cdef int _order_index, _ex_index
    cdef public bint traded
    
    cpdef add_order_to_history(self, dict order)
    cpdef add_order_to_book(self, int trader_id, int order_id, int timestamp, int quantity, Side side, int price)
    cdef void _remove_order(self, int trader_id, int order_id, int quantity)
    cdef void _modify_order(self, int trader_id, int order_id, int quantity)
    cdef void _add_trade_to_book(self, int resting_trader_id, int resting_order_id, int resting_timestamp,
                                 int incoming_trader_id, int incoming_order_id,
                                 int timestamp, int price, int quantity, Side side)
    cdef void _confirm_trade(self, int timestamp, Side order_side, int order_quantity, int order_id,
                             int order_price, int trader_id)
    cdef BookTop get_ask(self)
    cdef BookTop get_bid(self)
    cpdef process_order(self, dict order)
    cdef void _match_trade(self, int trader_id, int order_id, int timestamp, int quantity, Side side, int price)
    cpdef dict report_top_of_book(self, int now_time)