import simpy
from components.xpu import Xpu, Dtypes
from components.ccl import Ccl
from components.hps import Hps
from trace import collect, monitor, dump_perfetto
from functools import partial
from simpy.events import AllOf
from utils import *
from simpy.events import AllOf
from simpy.util import start_delayed
from models.tformer import vanilla_tformer_procs
from models.llama_mm import llama_im_txt_train

a100_specs = XpuSpecs((312000, 0.47), (1935, 0.7), (80, 0.85), "A100")
cluster_specs = ClusterSpecs(TP=96, DP=11, scale_up=600*GIGA, scale_out=100*GIGA, HB=8)
xpu_specs = a100_specs
opt175b = ModelSpecs(1000, 2048, 12288, 4*12288, 50272, 96, True, False, None, Dtypes.FP16, 1, 1, False, "opt175b")
opt175b_prod = TrainingSpecs(cluster_specs, a100_specs, opt175b)


if __name__ == "__main__":
    tspecs = [opt175b_prod]
    for tspec in tspecs:
        model_specs = tspec.model_specs
        cluster_specs = tspec.cluster_specs
        xpu_specs = tspec.xpu_specs
        for HB in [cluster_specs.HB, cluster_specs.TP]:
            cluster_specs.HB = HB
            env = simpy.Environment()
            data = []
            monitor2 = partial(monitor, data)
            collect(env, monitor2)

            env.process(vanilla_tformer_procs(env,
                                              xpu_specs,
                                              model_specs,
                                              cluster_specs))

    #vit_h14 = ModelSpecs(25000, 1, 1280, 4*1280, 1000, 32, True, False, VisionSpecs(224, 224, 14, 3), Dtypes.FP16, 1, 1)
    # section 7.4 pretraining - 336x336; global 16384
    #llama_vit = ModelSpecs(10000, 1, 1280, 4*1280, 128256, 40, True, False, VisionSpecs(224, 224, 14, 3), Dtypes.FP16, 1, 1)
    #llama_text = ModelSpecs(32, 8192, 8192, 3.5*8192, 128256, 4, True, True, None, Dtypes.FP16, 8, 64)
    #env.process(llama_im_txt_train(env, xpu_specs, llama_vit, llama_text, cluster_specs))
            env.run()
            del env
            total_xpus = 1*1
            xpus = [f"xpu{i}" for i in range(total_xpus)]
            trace_file_name = "_".join([model_specs.name,
                                        xpu_specs.name,
                                        f"{cluster_specs.TP}X{cluster_specs.DP}",
                                        f"HB_{HB}"
                                        ])

            dump_perfetto(["ccl", "hps", "xpu", "hbm"],
                          [[f"tp_comm{i}" for i in range(cluster_specs.DP)] + [f"dp_comm{i}" for i in range(1)],
                           ["read", "write"],
                           xpus,
                           ["ctr_mem_capacity"]],
                          data,
                          trace_file_name)

            # for d in data:
            #     evt = d[2]
            #     if isinstance(evt, simpy.events.Timeout) and evt.value is not None:
            #         print(d[0], evt.value)
            out_data = list(reversed(data))
            final_time = out_data[0][0]
            for d in out_data:
                evt = d[2]
                if isinstance(evt, simpy.events.Timeout) and evt.value is not None:
                    if isinstance(evt.value, EventData) and "ckpt" in "_".join(evt.value.name):
                        continue
                    else:
                        iter_time = d[0]
                        print("xpu:", final_time - iter_time, iter_time, evt.value, evt.value.name)
                        break
            print(model_specs.name, xpu_specs.name, cluster_specs.TP, cluster_specs.DP, cluster_specs.HB, iter_time)
            reset_cid()
