import simpy
from enum import Enum
from simpy.events import AllOf
from simpy.util import start_delayed
from typing import Tuple, List
from utils import *
from dataclasses import dataclass

class Xpu:
    def __init__(self, env, specs: XpuSpecs, dev_id: int):
        self.flops = [specs.fp32_gflops[0] *
                      specs.fp32_gflops[1] *
                      GIGA *
                      dtype.value for dtype in Dtypes]
        self.mem_bw = specs.mem_bw_g[0] * specs.mem_bw_g[1] * GIGA
        self.mem_cap = specs.mem_cap_g[0] * specs.mem_cap_g[1] * GIGA
        self.memory = simpy.Container(env, init=0, capacity=self.mem_cap)
        self.env = env
        self.tile_size = 128
        self.dev_id = dev_id
        self.cid = next_cid()
        self.mem_cap_cid = next_cid()
        self.mem_contents = {}
        self.flop_count = 0
        self.compute_time = 0

    def evt_data(self, name):
        return EventData(name, self.env.now, ComponentType.XPU, self.cid, self.dev_id)

    def compute(self, flops, dtype, op):
        self.flop_count += flops
        # print("FL:", dtype, self.flops[dtype.value-1]/10**12)
        comp_time = int((flops / self.flops[dtype.value - 1]) * MICRO)
        self.compute_time += comp_time
        yield self.env.timeout(comp_time, value=self.evt_data(op + ["compute"]))

    def mem_access(self, bts, is_read, dtype, op):
        rd_bytes = bts * dtype.byte_size()
        mem_time = int((bts / self.mem_bw) * MICRO)
        yield self.env.timeout(mem_time, value=self.evt_data(op + ["mem_rd" if is_read else "mem_wr"]))

    def matmul(self, b, m, n, p, dtype, ckpt, op):
        wr_size = b * m * p
        if ckpt:
            yield self.env.process(self.mem_fill(wr_size, dtype, op))
        is_read = True
        rd_size = b * m * n + b * n * p
        mem_rd = self.env.process(self.mem_access(rd_size, is_read, dtype, op))
        macs = b * m * n * p
        comp_proc = self.env.process(self.compute(macs*2, dtype, op))
        is_read = False
        mem_wr = self.env.process(self.mem_access(wr_size, is_read, dtype, op))
        yield AllOf(self.env, [mem_rd, comp_proc, mem_wr])

    def elem_mul(self, elems, dtype, ckpt, op):
        wr_size = elems
        if ckpt:
            yield self.env.process(self.mem_fill(wr_size, dtype, op))
        is_read = True
        rd_size = 2 * elems
        mem_rd = self.env.process(self.mem_access(rd_size, is_read, dtype, op))
        macs = elems
        comp_proc = self.env.process(self.compute(macs, dtype, op))
        is_read = False
        mem_wr = self.env.process(self.mem_access(wr_size, is_read, dtype, op))
        yield AllOf(self.env, [mem_rd, comp_proc, mem_wr])

    def elem_mul_bk(self, elems, dtype, op):
        emul = self.elem_mul(elems, dtype, False, op)
        yield self.env.process(self.mem_free(elems, dtype, op))

    def matmul_bk(self, b, m, n, p, g, dtype, op, free_act=True):
        # Assume inputs to matmul were saved
        inp = b * m * n
        out = b * m * p
        wt_grad = b * n * (p // g)
        wt_macs = b * n * m * p
        wt_rd_size = inp + out
        is_read = True
        # wt grad matmul
        wt_grad_op = op + ["wt_grad"]
        yield self.env.process(self.mem_fill(wt_grad, dtype, wt_grad_op))
        wt_rd = self.env.process(self.mem_access(wt_rd_size, is_read, dtype, wt_grad_op))
        wt_comp = self.env.process(self.compute(wt_macs * 2, dtype, wt_grad_op))

        is_read = False
        wt_wr = self.env.process(self.mem_access(wt_grad, is_read, dtype, wt_grad_op))
        yield AllOf(self.env, [wt_rd, wt_comp, wt_wr])
        # Wt update - Read weight and weight grad
        is_read = True
        yield self.env.process(self.mem_access(wt_grad + wt_grad, is_read, dtype, wt_grad_op))
        yield self.env.process(self.mem_access(3 * wt_grad, is_read, Dtypes.FP32, wt_grad_op))
        wt_grad_up = op + ["wt_grad_upd"]
        opt_upd = op + ["opt_upd"]
        yield self.env.timeout(1, value=self.evt_data(wt_grad_op))
        yield self.env.timeout(1, value=self.evt_data(opt_upd))

        is_read = False
        wt_upd = self.env.process(self.mem_access(wt_grad, is_read, dtype, wt_grad_up))
        yield wt_upd
        o_upd = self.env.process(self.mem_access(3 * wt_grad, is_read, Dtypes.FP32, opt_upd))
        yield o_upd

        # w_free = self.env.process(self.mem_free(wt_grad, dtype, wt_grad_op)) # Free w_grad
        # yield w_free

        l_grad = b * m * n
        l_macs = b * m * p * n
        l_rd_size = b * m * p + b * p * n
        is_read = True
        l_grad_op = op + ["l_grad"]
        #print("L_grad:", b, m, n, l_grad * 2 / GIGA)
        yield self.env.process(self.mem_fill(l_grad, dtype, l_grad_op))
        l_rd = self.env.process(self.mem_access(l_rd_size, is_read, dtype, l_grad_op))
        l_comp = self.env.process(self.compute(l_macs * 2, dtype, l_grad_op))

        is_read = False
        l_wr = self.env.process(self.mem_access(l_grad, is_read, dtype, l_grad_op))
        yield AllOf(self.env, [l_rd, l_comp, l_wr])

        w_free = self.env.process(self.mem_free(wt_grad, dtype, wt_grad_op)) # Free w_grad
        act_free = self.env.process(self.mem_free(inp, dtype, l_grad_op)) # Free act grad
        if free_act:
            out_free = self.env.process(self.mem_free(out, dtype, op)) # Free output loss grad
            yield AllOf(self.env, [out_free, w_free, act_free])
        else:
            yield AllOf(self.env, [w_free, act_free])


    @staticmethod
    def oom_msg(req, avail):
        return f"XPU - Requested({req / GIGA} GB) > Available({(avail)/GIGA} GB) not available"

    def mem_fill(self, size, dtype, op):
        size_in_bytes = size * dtype.byte_size()
        op_name = "_".join(op)
        # if "W_" in op_name:
        #     print("Counting ops:", op_name)
        #     self.param_count += size
        if "embed_load" in op_name:
            print("Embed params:", 2 * size / GIGA)
        if self.mem_contents.get(op_name, None) is None:
            self.mem_contents[op_name] = 1
        else:
            self.mem_contents[op_name] += 1
        if (self.memory.level + size_in_bytes) < self.memory.capacity:
            yield self.memory.put(size_in_bytes)
            yield self.env.timeout(1, value=self.evt_data(op + ["mem_fill"]))
            yield self.env.timeout(1, value=CounterData(self.memory.level / self.memory.capacity,
                                                        self.mem_cap_cid,
                                                        self.dev_id))
        else:
            raise Exception(Xpu.oom_msg(size_in_bytes, self.memory.capacity - self.memory.level))

    def mem_free(self, size, dtype, op):
        size_in_bytes = size * dtype.byte_size()
        op_name = "_".join(op)
        if self.mem_contents.get(op_name, None) is None:
            raise Exception(f"mem_free before alloc", op_name)
        else:
            self.mem_contents[op_name] -= 1
        if (self.memory.level - size_in_bytes) >= 0:
            yield self.memory.get(size_in_bytes)
            yield self.env.timeout(1, value=self.evt_data(op + ["mem_free"]))
        else:
            raise Exception(f"Free below zero. Current:{self.memory.level}|Free:{size_in_bytes}")

    def mem_rem(self):
        return (self.memory.capacity - self.memory.level) / GIGA
