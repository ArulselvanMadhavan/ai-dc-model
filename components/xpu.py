import simpy
from enum import Enum
from simpy.events import AllOf

class Dtypes(Enum):
    FP32 = 1
    FP16 = 2                    # FP16 = BF16 in Nvidia H100
    FP8 = 3

GIGA = 10**9
MILLI = 10**3

class Xpu:
    def __init__(self, env, fp32_gflops):
        self.flops = [fp32_gflops * GIGA * dtype.value for dtype in Dtypes]
        self.mem_bw = 3350 * GIGA
        self.mem_cap = 80
        self.env = env

    def compute(self, flops, dtype, op):
        comp_time = int((float(flops) / self.flops[dtype.value - 1]) * MILLI)
        yield self.env.timeout(comp_time, value=[op, "compute"])

    def mem_read(self, rd_bytes, op):
        mem_time = int((float(rd_bytes) / self.mem_bw) * MILLI)
        yield self.env.timeout(mem_time, value=[op, "mem_read"])

    def matmul(self, m, n, p, dtype=Dtypes.FP16):
        op = "matmul"
        flops = m * n * p
        rd_bytes = (m * n + n * p + m * p) * dtype.value
        mem_proc = self.env.process(self.mem_read(rd_bytes, op))
        comp_proc = self.env.process(self.compute(flops, dtype, op))
        yield AllOf(self.env, [mem_proc, comp_proc])
