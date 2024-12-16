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
from copy import deepcopy
import csv

a100_specs = XpuSpecs((312000, 0.47), (1935, 0.7), (80, 0.85), "A100")
h100_specs = XpuSpecs((989000, 0.43), (3350, 0.7), (80, 0.85), "H100")

opt175b_cluster_specs = ClusterSpecs(TP=96, DP=11, scale_up=600*GIGA, scale_out=100*GIGA, HB=8)
llama16k_cluster = ClusterSpecs(TP=8*16, DP=128, scale_up=900*GIGA,scale_out=100*GIGA, HB=8)

opt175b = ModelSpecs(1000, 2048, 12288, 4*12288, 50272, 96, True, False, None,
                     Dtypes.FP16, 1, 1, False, "opt175b", 140000)

llama_vit_h14 = ModelSpecs(16384, 1, 1280, 4*1280, 128256, 40, True, False, VisionSpecs(224, 224, 14, 3), Dtypes.FP16, 1, 1, False, "llama-vit-850M", 1)
llama70b_text = ModelSpecs(16*128, 8192, 8192, 3.5*8192, 128256, 80, True,
                           False, None, Dtypes.FP16, 8, 64, True, "llama70B", 975000)
llama70b_mm = ModelSpecs(16384, 8192, 8192, 3.5*8192, 128256, 80, True,
                         True, None, Dtypes.FP16, 8, 64, True, "llama70Bmm",
                         6*GIGA//16384)
# section 3.4.1 - 16M tokens
llama405b_text = ModelSpecs(16*128, 8192, 16384, 3.25*16384, 128256, 126, True,
                            False, None, Dtypes.FP16, 8, 128, True, "llama405B", 930000)
opt175b_prod = TrainingSpecs(deepcopy(opt175b_cluster_specs), a100_specs, opt175b)
llama405b_prod = TrainingSpecs(deepcopy(llama16k_cluster), h100_specs, llama405b_text)
llama70b_prod = TrainingSpecs(deepcopy(llama16k_cluster), h100_specs, llama70b_text)

#Mtraining
# llama image-text - global batch size 16384 - section 7.4 - 6B pairs
llama_90b = deepcopy(llama70b_text)
llama_90b.name = "llama90B"
llama90b_prod = MtrainingSpecs(
    deepcopy(llama16k_cluster),
    h100_specs,
    llama_90b,
    llama_vit_h14
)

if __name__ == "__main__":
    rows = []
    # columns = ["model_name", "metric", "value"]
    columns = ["model_name", "sku", "XPU", "TP", "DP", "high_bw_domain", "iter_time", "training_time"]
    rows.append(columns)
    tspecs = [opt175b_prod, llama405b_prod, llama70b_prod, llama90b_prod]
    for tspec in tspecs:
        model_specs = tspec.model_specs
        cluster_specs = tspec.cluster_specs
        xpu_specs = tspec.xpu_specs
        for HB in [cluster_specs.HB, cluster_specs.TP]:
            TP = cluster_specs.TP
            DP = cluster_specs.DP
            if HB == cluster_specs.TP:
                prefix = "Passage"
            else:
                prefix = "Nvidia"
            sku = f"{prefix}_{xpu_specs.name}_{TP*DP}({TP},{DP})"
            cluster_specs.HB = HB
            env = simpy.Environment()
            data = []
            monitor2 = partial(monitor, data)
            collect(env, monitor2)
            if isinstance(tspec, MtrainingSpecs):
                env.process(llama_im_txt_train(env, xpu_specs, tspec.sec_model, tspec.model_specs, cluster_specs))
            else:
                env.process(vanilla_tformer_procs(env, xpu_specs, model_specs, cluster_specs))

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
            row = [model_specs.name, sku, xpu_specs.name, cluster_specs.TP,
                  cluster_specs.DP, cluster_specs.HB, iter_time,
                  (iter_time * model_specs.num_iters) / (MICRO * 60*60*24)]
            rows.append(row)
            reset_cid()
    with open("results.csv", "w+") as f:
        w = csv.writer(f)
        w.writerows(rows)
