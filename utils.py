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

def reset_cid():
    global CID
    CID = 0
    return CID

CCL_ID = 0
def get_ccl_id():
    global CCL_ID
    return CCL_ID

def next_ccl_id():
    global CCL_ID
    result = CCL_ID
    CCL_ID = CCL_ID + 1
    return result

def reset_ccl_id():
    global CCL_ID
    CCL_ID = 0
    return CCL_ID


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

@dataclass
class CounterData:
    count: float
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

def calc_conv_out_dim(dim, padding, dilation, kernel_size, stride):
    """https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html"""
    res = (dim + (2 * padding) - dilation * (kernel_size - 1) - 1) // stride
    return res + 1

@dataclass
class VisionSpecs:
    H: int
    W: int
    P: int
    C: int

@dataclass
class ModelSpecs:
    G: int
    S: int
    E: int
    H: int
    V: int
    L: int
    is_train: bool
    freeze: bool
    vision: VisionSpecs
    param_dtype: Dtypes
    kv_heads: int
    num_heads: int
    is_llama_mlp: bool
    name: str
    num_iters: int

@dataclass
class ClusterSpecs:
    TP: int
    DP: int
    PP: int
    scale_up: int
    scale_out: int
    HB: int

@dataclass
class XpuSpecs:
    fp32_gflops: Tuple[int, float]
    mem_bw_g: Tuple[int, float]
    mem_cap_g: Tuple[int, float]
    name: str

@dataclass
class TrainingSpecs:
    cluster_specs: ClusterSpecs
    xpu_specs: XpuSpecs
    model_specs: ModelSpecs

@dataclass
class MtrainingSpecs(TrainingSpecs):
    # cluster_specs: ClusterSpecs
    # xpu_specs: XpuSpecs
    # base_model: ModelSpecs
    sec_model: ModelSpecs
