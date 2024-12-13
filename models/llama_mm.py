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
from dataclasses import dataclass


def load_qkv_ffns(env, L, E, E_tp, freeze, is_train, dtype, xpu, op):
    def add_opt_states(p, i, l):
        return [(xpu.mem_fill(p, Dtypes.FP32, op+[f"{k}_{i}_{l}"]))
                for k in ["momentum", "variance", "param"]]
    qkvs = []
    ffns = []
    for l in range(L):
        QKVs = ["Q", "K", "V"]
        qkvs = qkvs + [(xpu.mem_fill(E*E_tp, dtype, op+[f"W_{i}_{l}"])) for i in QKVs]
        FFNs = ["ffn1", "ffn2"]
        ffns = ffns + [(xpu.mem_fill(E*4*E_tp, dtype, op+[f"W_{i}_{l}"])) for i in FFNs]
        if is_train and freeze is False:
            os = [p for i in QKVs for p in add_opt_states(E*E_tp, i, l)]
            qkvs = qkvs + os
            os = [p for i in FFNs for p in add_opt_states(E*4*E_tp, i, l)]
            ffns = ffns + os
    return qkvs + ffns

def llama_im_txt_train(env, xpu_specs, im_model_specs, txt_model_specs, cluster_specs):
    TP = cluster_specs.TP
    DP = cluster_specs.DP
    bws = [cluster_specs.scale_up, cluster_specs.scale_out]
    bw_eff = [0.7, 0.7]
    has_bias = False
    HB = cluster_specs.HB
    tp_comms = [Ccl(env, [HB, TP//HB], bws, bw_eff) for i in range(DP)]
    dp_comm = Ccl(env, [TP, DP], bws, bw_eff)
    hps = Hps(env, hps_rd_bw=1000*GIGA, hps_wr_bw=500*GIGA)
    is_train = im_model_specs.is_train
    dev_id = 0
    xpu = Xpu(env, xpu_specs, dev_id)
    def im_model_run(model_specs):
        vision_specs = model_specs.vision
        H_out = calc_conv_out_dim(vision_specs.H, 0, 1, vision_specs.P, vision_specs.P)
        W_out = calc_conv_out_dim(vision_specs.W, 0, 1, vision_specs.P, vision_specs.P)
        S = H_out * W_out
        C = vision_specs.C
        K = vision_specs.P
        G = model_specs.G
        B = G // DP
        S = model_specs.S
        E = model_specs.E
        H = model_specs.H
        E_tp = model_specs.E // TP
        V = model_specs.V
        L = model_specs.L
        dtype = model_specs.param_dtype
        freeze = model_specs.freeze
        param_count = 4*E*E + 4*E + 2*E*H + E + H if has_bias else 4*E*E + 2*E*H
        assert E % TP == 0, "TP dim size should divide embedding dimension"
        freeze = model_specs.freeze
        print("im-model params:", (param_count * L + (6*E*E*6))/GIGA)
        kload = xpu.mem_fill(E_tp * vision_specs.C * vision_specs.P * vision_specs.P, dtype, ["conv_kernel_load"])
        qkv_ffns = load_qkv_ffns(env, L, E, E_tp, freeze, is_train, dtype, xpu, ["im"])
        return [kload] + qkv_ffns
        #yield AllOf(env, [kload, qkv_ffns])

    def txt_model_run(model_specs):
        G = model_specs.G
        B = G // DP
        S = model_specs.S
        E = model_specs.E
        assert E % TP == 0, "TP dim size should divide embedding dimension"
        H = model_specs.H
        E_tp = model_specs.E // TP
        V = model_specs.V
        L = model_specs.L
        bws = [cluster_specs.scale_up, cluster_specs.scale_out]
        bw_eff = [0.7, 0.7]
        dtype = Dtypes.FP16
        is_train = model_specs.is_train
        has_bias = False
        param_count = 4*E*E + 4*E + 3*E*H + E + H if has_bias else 4*E*E + 3*E*H
        out_params = 2 * E * V
        freeze = model_specs.freeze
        print("txt-model-params:", ((param_count * L)+out_params)/GIGA)
        eload = xpu.mem_fill(V*E_tp, dtype, ["embed_load"])
        qkv_ffns = load_qkv_ffns(env, L, E, E_tp, freeze, is_train, dtype, xpu, ["txt"])
        return [eload] + qkv_ffns


    im_model_load = im_model_run(im_model_specs)
    txt_model_load = txt_model_run(txt_model_specs)
    from typing import Generator
    all_loads = im_model_load + txt_model_load
    chunk_size = 10
    chunk = []
    for i, l in enumerate(all_loads):
        chunk.append(l)
        if i % chunk_size == 0:
            yield AllOf(env, [env.process(c) for c in chunk])
            chunk = []
            print("mem:", i, xpu.mem_rem())
    # yield AllOf(env, [env.process(ll) for ll in im_model_load])
    # print(xpu.mem_rem())
    # yield AllOf(env, all_loads)
