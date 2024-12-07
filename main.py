import simpy
from components.xpu import Xpu, Dtypes, XpuSpecs
from components.ccl import Ccl
from trace import trace, monitor
from functools import partial
from simpy.events import AllOf
from utils import GIGA, MICRO, MILLI
from simpy.events import AllOf
from simpy.util import start_delayed

if __name__ == "__main__":
    env = simpy.Environment()
    data = []
    monitor = partial(monitor, data)
    trace(env, monitor)

    xpu_specs = XpuSpecs((989000, 0.5), (3350, 0.7), (80, 0.85))

    TP = 2
    B = 32
    S = 512
    E = 12288
    E_tp = E // TP
    bws = [900 * GIGA, 100 * GIGA]
    dtype = Dtypes.FP16
    is_train = True
    tp_comm = Ccl(env, TP, bws)

    # TF graph on xpu
    def tformer(dev_id, L=1):
        def add_opt_states(p, i, l):
            return [env.process(xpu.mem_fill(p, Dtypes.FP32, f"{k}_{i}_{l}"))
                    for k in ["momentum", "variance", "param"]]

        qkvs = []
        ffns = []
        xpu = Xpu(env, xpu_specs, dev_id)
        for l in range(L):
            QKVs = ["Q", "K", "V"]
            qkvs = qkvs + [env.process(xpu.mem_fill(E*E_tp, dtype, f"W_{i}_{l}")) for i in QKVs]
            FFNs = ["ffn1", "ffn2"]
            ffns = ffns + [env.process(xpu.mem_fill(E*4*E_tp, dtype, f"W_{i}_{l}")) for i in FFNs]
            if is_train:
                os = [p for i in QKVs for p in add_opt_states(E*E_tp, i, l)]
                qkvs = qkvs + os
                os = [p for i in FFNs for p in add_opt_states(E*4*E_tp, i, l)]
                ffns = ffns + os

        yield AllOf(env, qkvs + ffns)
        yield env.process(xpu.mem_fill(B*S*E, dtype, "X"))
        for i in QKVs:
            yield env.process(xpu.matmul(1, B*S, E, E_tp, dtype, f"X@W_{i}"))
        # Ignore K^T
        yield env.process(xpu.matmul(B, S, E_tp, S, dtype, f"Q@K^T"))
        # Ignore softmax
        yield env.process(xpu.matmul(B, S, S, E_tp, dtype, f"A@V^T"))
        yield env.process(xpu.matmul(1, B*S, E_tp, E, dtype, f"out_proj"))
        # All reduce
        yield env.process(tp_comm.all_reduce(B*S*E, dtype, f"xpu{dev_id}-attn_partial"))
        # Ignore layer norm; dropout steps
        yield env.process(xpu.matmul(1, B*S, E, 4*E_tp, dtype, f"ffn1"))
        yield env.process(xpu.matmul(1, B*S, 4*E_tp, E, dtype, f"ffn2"))
        yield env.process(tp_comm.all_reduce(B*S*E, dtype, f"xpu{dev_id}-ffn_partial"))
        print(xpu.mem_rem())

    def tp_procs():
        yield AllOf(env, [start_delayed(env, tformer(i, L=1), i+1) for i in range(TP)])
    env.process(tp_procs())
    env.run()

    # for d in data:
    #     evt = d[2]
    #     if isinstance(evt, simpy.events.Timeout) and evt.value is not None:
    #         print(d[0], evt.value)
