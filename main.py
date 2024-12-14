import simpy
from components.xpu import Xpu, Dtypes, XpuSpecs
from components.ccl import Ccl
from components.hps import Hps
from trace import trace, monitor
from functools import partial
from simpy.events import AllOf
from utils import *
from simpy.events import AllOf
from simpy.util import start_delayed
from trace import dump_perfetto
from models.tformer import vanilla_tformer_procs
from models.llama_mm import llama_im_txt_train

if __name__ == "__main__":
    env = simpy.Environment()
    data = []
    monitor = partial(monitor, data)
    trace(env, monitor)

    a100_specs = XpuSpecs((312000, 0.47), (1935, 0.7), (80, 0.85))
    cluster_specs = ClusterSpecs(8, 8, 600*GIGA, 100*GIGA, 8)
    xpu_specs = a100_specs
    opt175b = ModelSpecs(1000, 2048, 12288, 4*12288, 50272, 1, True, False, None, Dtypes.FP16)
    model_specs = opt175b
    # env.process(vanilla_tformer_procs(env, xpu_specs, model_specs, cluster_specs))

    vit_h14 = ModelSpecs(25000, 1, 1280, 4*1280, 1000, 32, True, False, VisionSpecs(224, 224, 14, 3), Dtypes.FP16)
    llama_vit = ModelSpecs(1000, 1, 1280, 4*1280, 128256, 40, True, False, VisionSpecs(224, 224, 14, 3), Dtypes.FP16)
    llama_text = ModelSpecs(100, 8192, 8192, 3.5*8192, 128256, 80, True, True, None, Dtypes.FP16)
    env.process(llama_im_txt_train(env, xpu_specs, llama_vit, llama_text, cluster_specs))
    env.run()
    total_xpus = 1*1
    xpus = [f"xpu{i}" for i in range(total_xpus)]
    dump_perfetto(["ccl", "hps", "xpu", "hbm"],
                  [[f"tp_comm{i}" for i in range(cluster_specs.DP)] + [f"dp_comm{i}" for i in range(1)],
                   ["read", "write"],
                   xpus,
                   ["ctr_mem_capacity"]],
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
