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

# method to print the divisors
def get_divisors(n, start=1) :
    result = []
    i = start
    while i <= (n // (start*2)) :
        if (n % i==0) :
            result.append(i)
            #print (i,end=" ")
        i = i + 1
    return result

a100_specs = XpuSpecs((312000, 0.47), (1935, 0.7), (80, 0.85), "A100")
h100_specs = XpuSpecs((989000, 0.43), (3350, 0.7), (80, 0.85), "H100")

opt175b_cluster_specs = ClusterSpecs(TP=96, DP=11, PP=12, scale_up=600*GIGA, scale_out=100*GIGA, HB=8)
llama16k_cluster = ClusterSpecs(TP=8*16, DP=128, PP=1, scale_up=900*GIGA,scale_out=100*GIGA, HB=8)

opt175b = ModelSpecs(1000, 2048, 12288, 4*12288, 50272, 96, True, False, None,
                     Dtypes.FP16, 1, 1, False, "opt175b", 140000)

llama_vit_h14 = ModelSpecs(16384, 1, 1280, 4*1280, 128256, 40, True, False, VisionSpecs(224, 224, 14, 3), Dtypes.FP16, 1, 1, False, "llama-vit-850M", 1)
llama70b_text = ModelSpecs(15*128, 8192, 8192, 3.5*8192, 128256, 80, True,
                           False, None, Dtypes.FP16, 8, 64, True, "llama70B", 975000)
llama8b_text = ModelSpecs(15*128, 8192, 4096, 3.5*4096, 128256, 32, True, False, None, Dtypes.FP16, 8, 32, True, "llama8B", 975000)
llama70b_mm = ModelSpecs(16384, 8192, 8192, 3.5*8192, 128256, 80, True,
                         True, None, Dtypes.FP16, 8, 64, True, "llama70Bmm",
                         6*GIGA//16384)
# section 3.4.1 - 16M tokens
llama405b_text = ModelSpecs(15*128, 8192, 16384, 3.25*16384, 128256, 126, True,
                            False, None, Dtypes.FP16, 8, 128, True, "llama405B", 930000)
opt175b_prod = TrainingSpecs(deepcopy(opt175b_cluster_specs), a100_specs, opt175b)
llama405b_prod = TrainingSpecs(deepcopy(llama16k_cluster), h100_specs, llama405b_text)
llama70b_prod = TrainingSpecs(deepcopy(llama16k_cluster), h100_specs, llama70b_text)
#llama8b_prod = TrainingSpecs(deepcopy(llama16k_cluster), h100_specs, )
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
    columns = ["model_name", "sku", "XPU", "TP", "DP",
               "HB", "high_bw_domain", "iter_time", "training_time"]
    rows.append(columns)
    tspecs = [
        opt175b_prod,
        # llama405b_prod,
        # llama70b_prod,
        # llama90b_prod
    ]
    for tspec in tspecs:
        model_specs = tspec.model_specs
        cluster_specs = tspec.cluster_specs
        xpu_specs = tspec.xpu_specs
        TP_orig = cluster_specs.TP
        DP_orig = cluster_specs.DP
        HB_orig = cluster_specs.HB
        #divs = get_divisors(model_specs.E, start=8)
        divs = [TP_orig]
        for tp in divs:
            cluster_specs.TP = tp
            cluster_specs.DP = (TP_orig * DP_orig) // tp
            if cluster_specs.DP < 4:
                print("Skipping due to low DP:", cluster_specs.DP)
                continue
            for HB in [HB_orig, cluster_specs.TP]:
                TP = cluster_specs.TP
                DP = cluster_specs.DP
                PP = cluster_specs.PP
                num_xpus = TP * DP
                cluster_specs.HB = HB
                #sku = f"{xpu_specs.name}_XPUs({num_xpus})_(TP,DP)({TP},{DP})_HB({HB})"
                sku = f"({TP},{DP})"
                try:
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
                    total_xpus = 1*1*PP
                    xpus = [f"xpu-group-{i}" for i in range(total_xpus)]
                    hbms = [f"ctr_hbm-group-{i}" for i in range(total_xpus)]
                    xpu_hbms = []
                    for x, h in zip(xpus, hbms):
                        xpu_hbms.append(x)
                        xpu_hbms.append(h)

                    trace_file_name = "_".join([model_specs.name,
                                                xpu_specs.name,
                                                f"{cluster_specs.TP}X{cluster_specs.DP}",
                                                f"HB_{HB}"
                                                ])
                    dump_perfetto(["ccl", "hps", "xpuXhbm"],
                                  [[f"tp_comm{i}" for i in range(cluster_specs.DP)] +
                                   [f"dp_comm{i}" for i in range(1)] +
                                   [f"pp_comm{i}" for i in range(cluster_specs.PP)],
                                   ["read", "write"],
                                   xpu_hbms],
                                  data,
                                  trace_file_name)

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
                    if HB == TP:
                        hbw = f"PSG"
                    else:
                        hbw = f"DGX"
                    row = [model_specs.name, sku, xpu_specs.name, cluster_specs.TP,
                           cluster_specs.DP, HB, hbw, iter_time,
                          (iter_time * model_specs.num_iters) / (MICRO * 60*60*24)]
                    rows.append(row)
                    reset_cid()
                    reset_ccl_id()
                    print("Passing", TP, DP, TP * DP, HB)
                except Exception as e:
                    reset_cid()
                    reset_ccl_id()
                #     print("Except:", TP, DP, TP * DP, HB, e)
                #     if "XPU - Requested" in str(e):
                #         continue
                #     else:
                #         print(rows)
                #         continue
                    raise e

    with open("results.csv", "w+") as f:
        w = csv.writer(f)
        w.writerows(rows)
