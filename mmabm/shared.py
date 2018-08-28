from enum import Enum


class Side(Enum):
    BID = 1
    ASK = 2
    
    
class OType(Enum):
    ADD = 1
    CANCEL = 2
    MODIFY = 3
    
    
class TType(Enum):
    ZITrader = 0
    Provider = 1
    MarketMaker = 2
    PennyJumper = 3
    Taker = 4
    Informed = 5
    