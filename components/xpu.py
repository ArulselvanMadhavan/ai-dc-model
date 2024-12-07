import simpy
from enum import Enum
from simpy.events import AllOf
from simpy.util import start_delayed
from typing import Tuple, List
from utils import GIGA, MICRO, MILLI, EventData, Dtypes
from dataclasses import dataclass

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
        self.tile_size = 128

    def compute(self, flops, dtype, op):
        comp_time = int((flops / self.flops[dtype.value - 1]) * MICRO)
        yield self.env.timeout(comp_time, value=EventData([op, "compute"], self.env.now))

    def mem_access(self, bts, is_read, dtype, op):
        rd_bytes = bts * dtype.byte_size()
        mem_time = int((bts / self.mem_bw) * MICRO)
        yield self.env.timeout(mem_time, value=EventData([op, "mem_rd" if is_read else "mem_wr"], self.env.now))

    def matmul(self, b, m, n, p, dtype, op):
        wr_size = b * m * p

        yield self.env.process(self.mem_fill(wr_size, dtype, op))

        is_read = True
        rd_size = b * m * n + b * n * p
        mem_rd = self.env.process(self.mem_access(rd_size, is_read, dtype, op))

        macs = b * m * n * p
        comp_proc = self.env.process(self.compute(macs*2, dtype, op))

        is_read = False
        mem_wr = self.env.process(self.mem_access(wr_size, is_read, dtype, op))

        yield AllOf(self.env, [mem_rd, comp_proc, mem_wr])

    @staticmethod
    def oom_msg(req, avail):
        return f"XPU - Requested({req / GIGA} GB) > Available({(avail)/GIGA} GB) not available"

    def mem_fill(self, size, dtype, op):
        size_in_bytes = size * dtype.byte_size()
        if (self.memory.level + size_in_bytes) < self.memory.capacity:
            yield self.memory.put(size_in_bytes)
            yield self.env.timeout(1, value=EventData([op, "mem_fill"], self.env.now))
        else:
            raise Exception(Xpu.oom_msg(size_in_bytes, self.memory.capacity - self.memory.level))

    def mem_rem(self):
        return (self.memory.capacity - self.memory.level) / GIGA
