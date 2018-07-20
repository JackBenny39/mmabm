cdef class ZITrader:
    cdef public str trader_type
    cdef public int trader_id, quantity
    cdef int _quote_sequence
    cdef public list quote_collector
    
    cdef int _make_q(self, int maxq)
    cdef dict _make_add_quote(self, int time, str side, int price)
 
    
cdef class Provider(ZITrader):
    cdef double _delta
    cdef public list cancel_collector
    cdef public dict local_book
    
    cdef dict _make_cancel_quote(self, dict q, int time)
    cpdef confirm_cancel_local(self, dict cancel_dict)
    cpdef confirm_trade_local(self, dict confirm)
    cpdef bulk_cancel(self, int time)
    cpdef process_signal(self, int time, dict qsignal, double q_provider, double lambda_t)
    cdef int _choose_price_from_exp(self, str buysell, int inside_price, double lambda_t)
    
    
cdef class MarketMaker(Provider):
    cdef int _num_quotes, _quote_range, _position, _cash_flow
    cdef public list cash_flow_collector
    
    cpdef confirm_trade_local(self, dict confirm)
    cdef void _cumulate_cashflow(self, int timestamp)
    cpdef process_signal(self, int time, dict qsignal, double q_provider, double lambda_t)
    
    
cdef class PennyJumper(ZITrader):
    cdef int _mpi
    cdef public list cancel_collector
    cdef dict _ask_quote, _bid_quote
    
    cdef dict _make_cancel_quote(self, dict q, int time)
    cpdef confirm_trade_local(self, dict confirm)
    cpdef process_signal(self, int time, dict qsignal, double q_taker)
    
    
cdef class Taker(ZITrader):
    
    cpdef process_signal(self, int time, double q_taker)
    
    
cdef class InformedTrader(ZITrader):
    cdef str _side
    cdef int _price
    
    cpdef process_signal(self, int time, double q_taker)
    