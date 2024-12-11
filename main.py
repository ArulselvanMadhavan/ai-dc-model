import simpy
from components.xpu import Xpu, Dtypes, XpuSpecs
from components.ccl import Ccl
from components.hps import Hps
from trace import trace, monitor
from functools import partial
from simpy.events import AllOf
from utils import GIGA, MICRO, MILLI
from simpy.events import AllOf
from simpy.util import start_delayed
from trace import dump_perfetto
from models.tformer import vanilla_tformer_procs
if __name__ == "__main__":
    env = simpy.Environment()
    data = []
    monitor = partial(monitor, data)
    trace(env, monitor)

    #h100_specs = XpuSpecs((989000, 0.5), (3350, 0.7), (80, 0.85))
    a100_specs = XpuSpecs((312000, 0.47), (1935, 0.7), (80, 0.85))
    xpu_specs = a100_specs
    DP = 256
    TP = 128

    env.process(vanilla_tformer_procs(env, xpu_specs, TP, DP))
    env.run()
    total_xpus = 1 * 1
    xpus = [f"xpu{i}" for i in range(total_xpus)]
    dump_perfetto(["ccl", "hps", "xpu", "hbm"],
                  [[f"tp_comm{i}" for i in range(1)] + ["dp_comm"], ["read", "write"], xpus, ["ctr_mem_capacity"]],
                  data)

    # for d in data:
    #     evt = d[2]
    #     if isinstance(evt, simpy.events.Timeout) and evt.value is not None:
    #         print(d[0], evt.value)

    for d in reversed(data):
        evt = d[2]
        if isinstance(evt, simpy.events.Timeout) and evt.value is not None:
            print(d[0], evt.value)
            break
