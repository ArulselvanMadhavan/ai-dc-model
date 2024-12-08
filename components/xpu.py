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
    def __init__(self, env, specs: XpuSpecs, dev_id: int):
        self.flops = [specs.fp32_gflops[0] * specs.fp32_gflops[1] * GIGA * dtype.value for dtype in Dtypes]
        self.mem_bw = specs.mem_bw_g[0] * specs.mem_bw_g[1] * GIGA
        self.mem_cap = specs.mem_cap_g[0] * specs.mem_cap_g[1] * GIGA
        self.memory = simpy.Container(env, init=0, capacity=self.mem_cap)
        self.env = env
        self.tile_size = 128
        self.dev_id = dev_id

    def evt_data(self, name):
        return EventData(name, self.env.now, f"xpu_{self.dev_id}")

    def compute(self, flops, dtype, op):
        comp_time = int((flops / self.flops[dtype.value - 1]) * MICRO)
        yield self.env.timeout(comp_time, value=self.evt_data([op, "compute"]))

    def mem_access(self, bts, is_read, dtype, op):
        rd_bytes = bts * dtype.byte_size()
        mem_time = int((bts / self.mem_bw) * MICRO)
        yield self.env.timeout(mem_time, value=self.evt_data([op, "mem_rd" if is_read else "mem_wr"]))

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

    def matmul_bk(self, b, m, n, p, dtype, op):
        # Assume inputs to matmul were saved
        inp = b * m * n
        out = b * m * p
        wt_grad = b * n * p      #
        wt_macs = b * n * m * p
        wt_rd_size = inp + out
        is_read = True
        yield self.env.process(self.mem_fill(wt_grad, dtype, op))
        wt_rd = self.env.process(self.mem_access(wt_rd_size, is_read, dtype, op))
        wt_comp = self.env.process(self.compute(wt_macs * 2, dtype, op))

        is_read = False
        wt_wr = self.env.process(self.mem_access(wt_grad, is_read, dtype, op))
        yield AllOf(self.env, [wt_rd, wt_comp, wt_wr])
        # Wt update - Read weight and weight grad
        is_read = True
        yield self.env.process(self.mem_access(wt_grad + wt_grad, is_read, dtype, op))
        yield self.env.process(self.mem_access(3 * wt_grad, is_read, Dtypes.FP32, op))
        yield self.env.timeout(1, value=self.evt_data([op, "wt_grad_upd"]))
        yield self.env.timeout(1, value=self.evt_data([op, "opt_upd"]))

        is_read = False
        wt_upd = self.env.process(self.mem_access(wt_grad, is_read, dtype, op))
        yield wt_upd
        o_upd = self.env.process(self.mem_access(3 * wt_grad, is_read, Dtypes.FP32, op))
        yield o_upd

        l_grad = b * m * n
        l_macs = b * m * p * n
        l_rd_size = b * m * p + b * p * n
        is_read = True
        yield self.env.process(self.mem_fill(l_grad, dtype, op))
        l_rd = self.env.process(self.mem_access(l_rd_size, is_read, dtype, op))
        l_comp = self.env.process(self.compute(l_macs * 2, dtype, op))
        is_read = False
        l_wr = self.env.process(self.mem_access(l_grad, is_read, dtype, op))
        yield AllOf(self.env, [l_rd, l_comp, l_wr])
        out_free = self.env.process(self.mem_free(out, dtype, op)) # Free output loss grad
        w_free = self.env.process(self.mem_free(wt_grad, dtype, op)) # Free w_grad
        act_free = self.env.process(self.mem_free(inp, dtype, op)) # Free act grad
        yield AllOf(self.env, [out_free, w_free, act_free])


    @staticmethod
    def oom_msg(req, avail):
        return f"XPU - Requested({req / GIGA} GB) > Available({(avail)/GIGA} GB) not available"

    def mem_fill(self, size, dtype, op):
        size_in_bytes = size * dtype.byte_size()
        if (self.memory.level + size_in_bytes) < self.memory.capacity:
            yield self.memory.put(size_in_bytes)
            yield self.env.timeout(1, value=self.evt_data([op, "mem_fill"]))
        else:
            raise Exception(Xpu.oom_msg(size_in_bytes, self.memory.capacity - self.memory.level))

    def mem_free(self, size, dtype, op):
        size_in_bytes = size * dtype.byte_size()
        if (self.memory.level - size_in_bytes) >= 0:
            yield self.memory.get(size_in_bytes)
            yield self.env.timeout(1, value=self.evt_data([op, "mem_free"]))
        else:
            raise Exception(f"Free below zero. Current:{self.memory.level}|Free:{size_in_bytes}")

    def mem_rem(self):
        return (self.memory.capacity - self.memory.level) / GIGA
