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
from dataclasses import dataclass

@dataclass
class ModelSpecs:
    G: int
    S: int
    E: int
    H: int
    V: int
    L: int
    is_train: bool
    freeze: bool

@dataclass
class ClusterSpecs:
    TP: int
    DP: int
    scale_up: int
    scale_out: int

def vanilla_tformer_procs(env, xpu_specs, model_specs, cluster_specs):
    TP = cluster_specs.TP
    DP = cluster_specs.DP
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
    param_count = 4*E*E + 4*E + 2*E*H + E + H if has_bias else 4*E*E + 2*E*H
    gpus_per_box = TP
    tp_comm = Ccl(env, [gpus_per_box, TP//gpus_per_box], bws, bw_eff)
    dp_comm = Ccl(env, [TP, DP], bws, bw_eff)
    hps = Hps(env, hps_rd_bw=1000*GIGA, hps_wr_bw=500*GIGA)
    freeze = model_specs.freeze
    def tformer(dev_id, L=1):
        #tp_comm = tp_comms[dev_id // TP]
        def add_opt_states(p, i, l):
            return [env.process(xpu.mem_fill(p, Dtypes.FP32, [f"{k}_{i}"]))
                    for k in ["momentum", "variance", "param"]]

        qkvs = []
        ffns = []
        xpu = Xpu(env, xpu_specs, dev_id) # Every xpu takes up two cids
        # Weight load
        yield env.process(xpu.mem_fill(V*E_tp, dtype, ["embed_load"]))
        for l in range(L):
            QKVs = ["Q", "K", "V"]
            qkvs = qkvs + [env.process(xpu.mem_fill(E*E_tp, dtype, [f"W_{i}"])) for i in QKVs]
            FFNs = ["ffn1", "ffn2"]
            ffns = ffns + [env.process(xpu.mem_fill(E*4*E_tp, dtype, [f"W_{i}"])) for i in FFNs]
            if is_train:
                os = [p for i in QKVs for p in add_opt_states(E*E_tp, i, l)]
                qkvs = qkvs + os
                os = [p for i in FFNs for p in add_opt_states(E*4*E_tp, i, l)]
                ffns = ffns + os
        yield AllOf(env, qkvs + ffns)
        yield env.process(xpu.mem_fill(E * (V // TP), dtype, ["vocab_load"]))
        yield env.process(xpu.mem_fill(B*S*E, dtype, ["X"]))
        yield env.process(xpu.matmul(1, B*S, V, E_tp, dtype, True, [f"X@emb"]))
        yield env.process(tp_comm.all_gather(B*S*E_tp, dtype, [f"xpu{dev_id}-embed-gather"]))

        # # Forward pass
        def fwd_pass(ckpt):
            yield env.process(xpu.mem_fill(B*S*E_tp, dtype, [f"xpu_{dev_id}-act-ckpt"]))
            for i in QKVs:
                yield env.process(xpu.matmul(1, B*S, E, E_tp, dtype, ckpt, [f"X@W_{i}"]))
            yield env.process(xpu.matmul(B, S, E_tp, S, dtype, ckpt, [f"Q@K^T"]))
            yield env.process(xpu.matmul(B, S, S, E_tp, dtype, ckpt, [f"A@V^T"]))
            yield env.process(xpu.matmul(1, B*S, E_tp, E, dtype, ckpt, [f"out_proj"]))
            yield env.process(tp_comm.all_reduce(B*S*E, dtype, [f"xpu{dev_id}-attn_partial"]))
            yield env.process(xpu.matmul(1, B*S, E, 4*E_tp, dtype, ckpt, [f"ffn1"]))
            yield env.process(xpu.matmul(1, B*S, 4*E_tp, E, dtype, ckpt, [f"ffn2"]))
            yield env.process(tp_comm.all_reduce(B*S*E, dtype, [f"xpu{dev_id}-ffn_partial"]))
            if ckpt is False:
                yield env.process(xpu.mem_free(B*S*E_tp, dtype, [f"xpu_{dev_id}-act-ckpt"]))

        def bk_pass(freeze):
            if freeze:
                # If freeze is set to true. No need to recompute activations.
                # Just run matmul in reverse order
                ckpt = False
                yield env.process(xpu.matmul(1, B*S, 4*E_tp, E, dtype, ckpt, [f"ffn2"]))
                yield env.process(xpu.matmul(1, B*S, E, 4*E_tp, dtype, ckpt, [f"ffn1"]))
                yield env.process(tp_comm.all_reduce(B*S*E, dtype, [f"xpu{dev_id}-ffn_grad"]))
                yield env.process(xpu.matmul(1, B*S, E_tp, E, dtype, ckpt, [f"out_proj"]))
                yield env.process(xpu.matmul(B, S, S, E_tp, dtype, ckpt, [f"A@V^T"]))
                yield env.process(xpu.matmul(B, S, E_tp, S, dtype, ckpt, [f"Q@K^T"]))
                for i in QKVs:
                    yield env.process(xpu.matmul(1, B*S, E, E_tp, dtype, ckpt, [f"X@W_{i}"]))
                yield env.process(tp_comm.all_reduce(B*S*E, dtype, [f"xpu{dev_id}-attn_ip_grad"]))
            else:
                yield env.process(fwd_pass(ckpt=True)) # no need to recompute?
                yield env.process(xpu.matmul_bk(1, B*S, 4*E_tp, E, dtype, [f"ffn2"]))
                yield env.process(xpu.matmul_bk(1, B*S, E, 4*E_tp, dtype, [f"ffn1"]))
                yield env.process(tp_comm.all_reduce(B*S*E, dtype, [f"xpu{dev_id}-ffn_grad"]))
                yield env.process(xpu.matmul_bk(1, B*S, E_tp, E, dtype, [f"out_proj"]))
                yield env.process(xpu.matmul_bk(B, S, S, E_tp, dtype, [f"A@V^T"]))
                yield env.process(xpu.matmul_bk(B, S, E_tp, S, dtype, [f"Q@K^T"]))
                for i in QKVs:
                    yield env.process(xpu.matmul_bk(1, B*S, E, E_tp, dtype, [f"X@W_{i}"]))
                yield env.process(tp_comm.all_reduce(B*S*E, dtype, [f"xpu{dev_id}-attn_ip_grad"]))
                yield env.process(xpu.mem_free(B*S*E_tp, dtype, [f"xpu_{dev_id}-act-ckpt"]))

        for l in range(L):
            yield env.process(fwd_pass(ckpt=False))
        # produce output
        yield env.process(xpu.matmul(1, B*S, E, (V // TP), dtype, True, [f"X@vocab"]))
        yield env.process(tp_comm.all_gather(B*S*(V // TP), dtype, [f"xpu{dev_id}-vocab_out-gather"]))
        yield env.process(dp_comm.all_reduce(G*S*V, dtype, [f"xpu_{dev_id}_loss_grad_partial"]))
        yield env.process(xpu.matmul_bk(1, B*S, E, (V // TP), dtype, [f"X@vocab"]))
        yield env.process(tp_comm.all_gather(B*S*(V // TP), dtype, [f"xpu{dev_id}-bk_vocab_out-gather"]))
        for l in range(L):
            yield env.process(bk_pass(freeze=freeze))

        yield env.process(xpu.matmul_bk(1, B*S, V, E_tp, dtype, [f"X@emb"]))
        yield env.process(tp_comm.all_gather(B*S*E_tp, dtype, [f"xpu{dev_id}-bk-embed-gather"]))

        yield env.process(hps.write(param_count / (DP * TP), dtype, [f"xpu{dev_id}_wt_ckpt"]))
        yield env.process(hps.write(3 * param_count / (DP * TP), Dtypes.FP32, [f"xpu{dev_id}_opt_ckpt"]))

        print("Free mem:", xpu.mem_rem())
        for k, v in xpu.mem_contents.items():
            if v > 0:
                print(k, v)

    yield AllOf(env, [start_delayed(env, tformer(i, L=96), i+1) for i in range(1*1)])
