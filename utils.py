from dataclasses import dataclass
from typing import Tuple, List
from enum import Enum

GIGA = 10**9
MILLI = 10**3
MICRO = 10**6
CID = 0

def get_cid():
    global CID
    return CID

def next_cid():
    global CID
    result = CID
    CID = CID + 1
    return result

class ComponentType(Enum):
    XPU = 1
    CCL = 2
    HPS = 3

@dataclass
class EventData:
    name: List[str]
    start_time: int
    cty: ComponentType
    cid: int
    tid: int

class Dtypes(Enum):
    FP32 = 1
    FP16 = 2                    # FP16 = BF16 in Nvidia H100
    FP8 = 3

    def byte_size(self):
        out = 0
        match self:
            case Dtypes.FP32:
                out = 4
            case Dtypes.FP16:
                out = 2
            case Dtypes.FP8:
                out = 1
        return out
