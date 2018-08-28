cdef enum Side:
    BID = 1
    ASK = 2
    
    
cdef enum OType:
    ADD = 1
    CANCEL = 2
    MODIFY = 3
    
    
cdef enum TType:
    ZITrader = 0
    Provider = 1
    MarketMaker = 2
    PennyJumper = 3
    Taker = 4
    Informed = 5