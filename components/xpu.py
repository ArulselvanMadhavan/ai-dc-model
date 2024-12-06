import simpy
from enum import Enum
from simpy.events import AllOf
from dataclasses import dataclass
from typing import Tuple

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

GIGA = 10**9
MILLI = 10**3
MICRO = 10**6

@dataclass
class XpuSpecs:
    fp32_gflops: Tuple[int, float]
    mem_bw_g: Tuple[int, float]
    mem_cap_g: Tuple[int, float]

class Xpu:
    def __init__(self, env, specs: XpuSpecs):
        self.flops = [specs.fp32_gflops[0] * specs.fp32_gflops[1] * GIGA * dtype.value for dtype in Dtypes]
        self.mem_bw = specs.mem_bw_g[0] * specs.mem_bw_g[1] * GIGA
        self.mem_cap = specs.mem_cap_g[0] * specs.mem_cap_g[1] * GIGA
        self.memory = simpy.Container(env, init=0, capacity=self.mem_cap)
        self.env = env

    def compute(self, flops, dtype, op):
        comp_time = int((flops / self.flops[dtype.value - 1]) * MICRO)
        print(comp_time)
        yield self.env.timeout(comp_time, value=[op, "compute"])

    def mem_access(self, rd_bytes, dtype, op):
        rd_bytes = rd_bytes * dtype.byte_size()
        mem_time = int((rd_bytes / self.mem_bw) * MICRO)
        yield self.env.timeout(mem_time, value=[op, "mem_access"])

    def matmul(self, m, n, p, dtype=Dtypes.FP16):
        op = "matmul"
        flops = m * n * p
        wr_size = m * p
        rd_size = m * n + n * p

        yield self.env.process(self.mem_fill(wr_size, dtype, "mat_out"))

        mem_proc = self.env.process(self.mem_access(rd_size + wr_size, dtype, op))
        comp_proc = self.env.process(self.compute(flops, dtype, op))
        yield AllOf(self.env, [mem_proc, comp_proc])

    @staticmethod
    def oom_msg(req, avail):
        return f"XPU - Requested({req / GIGA} GB) > Available({(avail)/GIGA} GB) not available"

    def mem_fill(self, size, dtype, op):
        size_in_bytes = size * dtype.byte_size()
        if (self.memory.level + size_in_bytes) < self.memory.capacity:
            yield self.memory.put(size_in_bytes)
            yield self.env.timeout(1, value=[op, "mem_fill"])
        else:
            raise Exception(Xpu.oom_msg(size_in_bytes, self.memory.capacity - self.memory.level))
