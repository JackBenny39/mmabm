cdef enum Side:
    BID = 1
    ASK = 2
    
    
cdef enum OType:
    ADD = 1
    CANCEL = 2
    MODIFY = 3