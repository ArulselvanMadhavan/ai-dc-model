import simpy
from components.xpu import Xpu, Dtypes, XpuSpecs
from components.ccl import Ccl
from components.hps import Hps
from functools import partial
from simpy.events import AllOf
from utils import *
from simpy.events import AllOf
from simpy.util import start_delayed
from dataclasses import dataclass
import numpy as np

def loss_gradient(env, G, B, S, E, V, TP, dtype, tp_comm, dp_comm, xpu, dev_id):
    yield env.process(xpu.matmul(1, B*S, E, (V // TP), dtype, True, [f"X@vocab"]))
    yield env.process(tp_comm.all_gather(B*S*V, dtype, [f"xpu{dev_id}-vocab_out-gather"]))

    yield env.process(dp_comm.all_reduce(B*S*V, dtype, [f"xpu_{dev_id}_loss_grad_partial"]))
    yield env.process(xpu.matmul_bk(1, B*S, E, (V // TP), 1, dtype, [f"X@vocab"]))
    yield env.process(tp_comm.all_gather(B*S*V, dtype, [f"xpu{dev_id}-bk_vocab_out-gather"]))

def load_qkv_ffns(env, L, E, E_tp, H_tp, num_heads, kv_heads,
                  is_llama_ffn, freeze, is_train, dtype, xpu, op):
    param_count = 0
    def add_opt_states(p, i, l):
        return [(xpu.mem_fill(p, Dtypes.FP32, op+[f"{k}_{i}"]))
                for k in ["momentum", "variance", "param"]]
    qkvs = []
    ffns = []

    for l in range(L):
        param_count += E*E_tp
        qkvs = qkvs + [(xpu.mem_fill(E*E_tp, dtype, op+[f"W_{i}"])) for i in ["Q"]]
        qkvs = qkvs + [(xpu.mem_fill(int(E*E_tp * kv_heads/num_heads),
                                     dtype, op+[f"W_{i}"])) for i in ["K", "V"]]
        qkvs = qkvs + [xpu.mem_fill(E*E_tp, dtype, op+[f"W_out"])]
        param_count += 2*E*E_tp*kv_heads/num_heads
        param_count += E*E_tp
        FFNs = ["ffn1", "ffn2", "ffn3"] if is_llama_ffn else ["ffn1", "ffn2"]
        ffns = ffns + [(xpu.mem_fill(E*H_tp, dtype, op+[f"W_{i}"])) for i in FFNs]
        param_count += 3*E*H_tp if is_llama_ffn else 2*E*H_tp
        if is_train and freeze is False:
            os = [p for i in ["Q"] for p in add_opt_states(E*E_tp, i, l)]
            qkvs = qkvs + os
            os = [p for i in ["K", "V"] for p in add_opt_states(int(E*E_tp*kv_heads/num_heads), i, l)]
            os = [p for i in FFNs for p in add_opt_states(E*H_tp, i, l)]
            ffns = ffns + os
    # print("Param Count per device:", param_count)
    return qkvs + ffns

def embed_fwd(env, B, S, E, V, E_tp, dtype, freeze, dev_id, tp_comm, xpu):
    yield env.process(xpu.matmul(1, B*S, V, E_tp, dtype, not freeze, [f"X@emb"]))
    yield env.process(tp_comm.all_gather(B*S*E, dtype, [f"xpu{dev_id}-embed-gather"]))

def img_emb_fwd(env, B, H_out, W_out, C, K, E, E_tp, dtype, freeze, dev_id, tp_comm, xpu):
    yield env.process(xpu.matmul(1, B*H_out*W_out, C*K*K, E_tp, dtype, not freeze, ["img-emb-gen"]))
    yield env.process(tp_comm.all_gather(B*H_out*W_out*E, dtype,[f"img-emb-gather"]))

def fwd_pass(env, B, S, E, V, E_tp, H_tp, dtype, num_heads, kv_heads,
             is_llama_mlp, freeze, dev_id, tp_comm, xpu, ckpt, op):
    yield env.process(xpu.mem_fill(B*S*E_tp, dtype, op + [f"xpu_{dev_id}-act-ckpt"]))
    for i in ["Q"]:
        yield env.process(xpu.matmul(1, B*S, E, E_tp, dtype, ckpt, op + [f"X@W_{i}"]))
    g = num_heads / kv_heads
    h = kv_heads
    assert E_tp % g == 0, f"E_tp{E_tp} not divisible by g{g}"
    for i in ["K", "V"]:
        yield env.process(xpu.matmul(1, B*S, E, E_tp // g, dtype, ckpt, op + [f"X@W_{i}"]))
        # Repeat kv to grow E_tp // g to E_tp
        # https://github.com/meta-llama/llama/blob/4d92db8a1db6c7f663252bf3477d2c4b8bad2385/llama/model.py#L171
    yield env.process(xpu.matmul(B, S, E_tp, S, dtype, ckpt, op + [f"Q@K^T"]))
    yield env.process(xpu.matmul(B, S, S, E_tp, dtype, ckpt, op + [f"A@V^T"]))
    yield env.process(xpu.matmul(1, B*S, E_tp, E, dtype, ckpt, op + [f"out_proj"]))
    yield env.process(tp_comm.all_reduce(B*S*E, dtype, op + [f"xpu{dev_id}-attn_partial"]))
    yield env.process(xpu.matmul(1, B*S, E, H_tp, dtype, ckpt, op + [f"ffn1"]))
    if is_llama_mlp:
        yield env.process(xpu.matmul(1, B*S, E, H_tp, dtype, ckpt, op + [f"gate_ffn"]))
        yield env.process(xpu.elem_mul(B*S*H_tp, dtype, ckpt, op + [f"elem_mul"]))
    yield env.process(xpu.matmul(1, B*S, H_tp, E, dtype, ckpt, op + [f"ffn2"]))
    yield env.process(tp_comm.all_reduce(B*S*E, dtype, op + [f"xpu{dev_id}-ffn_partial"]))
    if ckpt is False:
        yield env.process(xpu.mem_free(B*S*E_tp, dtype, op + [f"xpu_{dev_id}-act-ckpt"]))


def bk_pass(env, B, S, E, V, E_tp, H_tp, dtype, num_heads, kv_heads,
            is_llama_mlp, freeze, dev_id, tp_comm, xpu, op):
    g = num_heads / kv_heads
    if freeze:
        ckpt = False
        yield env.process(xpu.matmul(1, B*S, H_tp, E, dtype, ckpt, [f"ffn2"]))
        if is_llama_mlp:
            yield env.process(xpu.matmul(1, B*S, E, H_tp, dtype, ckpt, op + [f"gate_ffn"]))
            yield env.process(xpu.elem_mul(B*S*H_tp, dtype, ckpt, op + [f"elem_mul"]))
        yield env.process(xpu.matmul(1, B*S, E, H_tp, dtype, ckpt, [f"ffn1"]))
        yield env.process(tp_comm.all_reduce(B*S*E, dtype, [f"xpu{dev_id}-ffn_grad"]))
        yield env.process(xpu.matmul(1, B*S, E_tp, E, dtype, ckpt, [f"out_proj"]))
        yield env.process(xpu.matmul(B, S, S, E_tp, dtype, ckpt, [f"A@V^T"]))
        yield env.process(xpu.matmul(B, S, E_tp, S, dtype, ckpt, [f"Q@K^T"]))
        for i in ["K", "V"]:
            # Insert repeat kv to grow E_tp/g to E_tp
            yield env.process(xpu.matmul(1, B*S, E, E_tp, dtype, ckpt, op + [f"X@W_{i}"]))
        for i in ["Q"]:
            yield env.process(xpu.matmul(1, B*S, E, E_tp, dtype, ckpt, [f"X@W_{i}"]))
        yield env.process(tp_comm.all_reduce(B*S*E, dtype, [f"xpu{dev_id}-attn_ip_grad"]))
    else:
        yield env.process(fwd_pass(env, B, S, E, V, E_tp, H_tp, dtype,
                                   num_heads, kv_heads, is_llama_mlp, freeze, dev_id,
                                   tp_comm, xpu, ckpt=True, op=op))
        yield env.process(xpu.matmul_bk(1, B*S, H_tp, E, 1, dtype, op+[f"ffn2"]))
        if is_llama_mlp:
            yield env.process(xpu.matmul_bk(1, B*S, E, H_tp, 1, dtype, op + [f"gate_ffn"]))
            yield env.process(xpu.elem_mul_bk(B*S*H_tp, dtype, op=op + [f"elem_mul"]))
        yield env.process(xpu.matmul_bk(1, B*S, E, H_tp, 1, dtype, op+[f"ffn1"]))
        yield env.process(tp_comm.all_reduce(B*S*E, dtype, op+[f"xpu{dev_id}-ffn_grad"]))
        yield env.process(xpu.matmul_bk(1, B*S, E_tp, E, 1, dtype, op+[f"out_proj"]))
        yield env.process(xpu.matmul_bk(B, S, S, E_tp, g, dtype, op+[f"A@V^T"]))
        yield env.process(xpu.matmul_bk(B, S, E_tp, S, 1, dtype, op+[f"Q@K^T"]))
        for i in ["K", "V"]:
            yield env.process(xpu.matmul_bk(1, B*S, E, E_tp // g, 1, dtype, op + [f"X@W_{i}"]))
        for i in ["Q"]:
            yield env.process(xpu.matmul_bk(1, B*S, E, E_tp, 1, dtype, op+[f"X@W_{i}"]))
        yield env.process(tp_comm.all_reduce(B*S*E, dtype, op+[f"xpu{dev_id}-attn_ip_grad"]))
        yield env.process(xpu.mem_free(B*S*E_tp, dtype, op+[f"xpu_{dev_id}-act-ckpt"]))


def vanilla_tformer_procs(env, xpu_specs, model_specs, cluster_specs):
    TP = cluster_specs.TP
    DP = cluster_specs.DP
    PP = cluster_specs.PP
    G = model_specs.G
    for pp, p in [("TP", TP), ("DP", DP), ("PP", PP)]:
        assert p > 0, f"{pp}({p}) should be > 0"
    B = G // DP
    S = model_specs.S
    E = model_specs.E
    assert E % TP == 0, f"TP dim-{TP} size should divide embedding dimension-{E}"
    H = model_specs.H
    H_tp = H // TP
    E_tp = E // TP
    V = model_specs.V
    L = model_specs.L
    bws = [cluster_specs.scale_up, cluster_specs.scale_out]
    bw_eff = [0.7, 0.7]
    dtype = model_specs.param_dtype
    is_train = model_specs.is_train
    has_bias = False
    num_heads = model_specs.num_heads
    kv_heads = model_specs.kv_heads
    is_llama_mlp = model_specs.is_llama_mlp
    num_iters = model_specs.num_iters
    attn_params = 2*E*E + 2*E*E*(kv_heads/num_heads)
    mlp_params = 3 * E * H if is_llama_mlp else 2 * E * H
    out_params = 2 * E * V
    param_count = attn_params + 4*E + mlp_params + E + H if has_bias else attn_params + mlp_params
    total_params = (param_count * L) + out_params
    param_count_per_dev = total_params/(TP * PP)
    print("Total params:", TP, 3*E*E_tp, param_count / GIGA, total_params/GIGA)
    print("Max Params per device:", total_params/(GIGA * TP * PP),
          (param_count * L) / (GIGA * TP * PP))
    HB = cluster_specs.HB
    # tp comms
    tp_comms = [Ccl(env, [HB, TP//HB], bws, bw_eff) for i in range(DP * PP)]
    dp_comm_bws = [0, bws[1] * (TP // HB)]
    dp_comm = Ccl(env, [1, DP], dp_comm_bws, bw_eff)
    ll = L // PP
    # Only ever has one destination and one sender
    pp_comm_bws = [bws[0], bws[1] * (TP // HB)]
    pp_comms = [Ccl(env, [1, 1], pp_comm_bws, bw_eff) for i in range(PP - 1)]
    hps = Hps(env, hps_rd_bw=1000*GIGA, hps_wr_bw=500*GIGA)

    freeze = model_specs.freeze
    is_vision = model_specs.vision is not None
    is_text = not is_vision

    if is_vision:
        vision_specs = model_specs.vision
        H_out = calc_conv_out_dim(vision_specs.H, 0, 1, vision_specs.P,
                                  vision_specs.P)
        W_out = calc_conv_out_dim(vision_specs.W, 0, 1, vision_specs.P,
                                  vision_specs.P)
        S = H_out * W_out
        C = vision_specs.C
        K = vision_specs.P

    def load_params(pp_group_id, xpu, L, B):
        def add_opt_states(p, i, l):
            return [env.process(xpu.mem_fill(p, Dtypes.FP32, [f"{k}_{i}"]))
                    for k in ["momentum", "variance", "param"]]

        qkvs = []
        ffns = []
        # Weight load
        if pp_group_id == 0:
            if is_text:
                yield env.process(xpu.mem_fill((V//TP)*E, dtype, ["embed_load"]))
            elif is_vision:
                yield env.process(xpu.mem_fill(E_tp * vision_specs.C *
                                               vision_specs.P * vision_specs.P,
                                               dtype, ["conv_kernel_load"]))

        # weight load
        qkv_ffns = load_qkv_ffns(env, L, E, E_tp, H_tp, num_heads, kv_heads,
                                 is_llama_mlp, freeze, is_train, dtype, xpu, [])
        chunk_size = 3
        for l in range(0, len(qkv_ffns), chunk_size):
            qkv_procs = [env.process(l) for l in qkv_ffns[l:l+chunk_size]]
            yield AllOf(env, qkv_procs)

        if pp_group_id == PP - 1:
            if is_text:
                yield env.process(xpu.mem_fill(E * (V // TP), dtype, ["vocab_load"]))

    def load_act(pp_group_id, xpu, B):
        if is_text:
            yield env.process(xpu.mem_fill(B*S*E, dtype, ["X_txt"]))
        elif is_vision:
            yield env.process(xpu.mem_fill(B*H_out*W_out*C*K*K), ["X_img"])


    def tformer_fwd(pp_group_id, xpu, L=1, B=1):
        dev_id = 0
        tp_idx = 0 * DP + pp_group_id
        tp_comm = tp_comms[tp_idx]
        # fill space for activation tensor - Reusable
        if pp_group_id == 0:
            if is_text:
                yield env.process(xpu.mem_fill(B*S, dtype, ["raw_tokens_txt"]))
                yield env.process(embed_fwd(env, B, S, E, V, E_tp,
                                        dtype, freeze, dev_id, tp_comm, xpu))
            elif is_vision:
                yield env.process(xpu.mem_fill(B*S*H*W, dtype, ["raw_tokens_img"]))
                yield env.process(img_emb_fwd(env, B, H_out, W_out, C, K, E, E_tp,
                                          dtype, freeze, dev_id, tp_comm, xpu))


        #pp_comm = pp_comms[pp_group_id]
        for l in range(L):
            yield env.process(fwd_pass(env, B, S, E, V, E_tp, H_tp, dtype,
                                       num_heads, kv_heads, is_llama_mlp, freeze,
                                       dev_id, tp_comm, xpu, ckpt=False, op=[]))

    def tformer_bk(pp_group_id, xpu, L=1, B=1):
        dev_id = 0
        tp_idx = 0 * DP + pp_group_id
        tp_comm = tp_comms[tp_idx]

        for l in range(L):
            yield env.process(bk_pass(env, B, S, E, V, E_tp, H_tp, dtype,
                                      num_heads, kv_heads, is_llama_mlp, freeze, dev_id,
                                      tp_comm, xpu, op=[]))

        if pp_group_id == 0:
            if is_text and (not freeze):
                # all-scatter
                yield env.process(tp_comm.all_gather(B*S*E, dtype, [f"xpu{dev_id}-bk-embed-gather"]))
                yield env.process(xpu.matmul_bk(1, B*S, (V//TP), E, 1, dtype, [f"X@emb"], free_act=False))
            if is_vision and (not freeze):
                yield env.process(tp_comm.all_gather(B*H_out*W_out*E, dtype,[f"img-emb-gather"]))
                yield env.process(xpu.matmul_bk(1, B*H_out*W_out, C*K*K, E_tp, 1, dtype, ["image-embed-gen"], free_act=False))
        # yield env.process(hps.write(param_count / (DP * TP), dtype, [f"xpu{dev_id}_wt_ckpt"]))
        # yield env.process(hps.write(3 * param_count / (DP * TP), Dtypes.FP32, [f"xpu{dev_id}_opt_ckpt"]))


    # For every batch in m
    # go through fwd pass
    # initiate pp_comm
    # wait_until_done
    # reverse layers
    M = [B // PP] * PP
    xpus = []
    dev_id = 0
    for pp in range(PP):
        xpu = Xpu(env, xpu_specs, dev_id) # Every xpu takes up two cids
        yield env.process(load_params(pp, xpu, L=ll, B=M[0]))
        yield env.process(load_act(pp, xpu, B=M[0]))
        xpus.append(xpu)

    arr = np.empty([PP, len(M) + PP - 1], dtype=object)
    # pp_arr = np.empty([len(M), PP], dtype=object)
    for m_id, m in enumerate(M):
        for pp in range(PP):
            xpu = xpus[pp]
            arr[pp][m_id] = tformer_fwd(pp, xpu, L=ll, B=m)
            # print("Free mem:", m_id, pp, xpu.mem_rem())
            # for k, v in xpu.mem_contents.items():
            #     if pp == 0 and v > 0:
            #         print(k, v)
    def schedule_procs(arr, roll_fn):
        PP = arr.shape[0]
        for pp in range(PP):
            arr[pp] = np.roll(arr[pp], roll_fn(PP, pp))
        arr = np.transpose(arr, [1, 0])
        for m_id in range(arr.shape[0]):
            row = arr[m_id]
            procs = [env.process(e) for e in row.tolist() if e is not None]
            pp_procs = [env.process(pp_comms[e_idx].send(m * S * E, dtype, op=[]))
                        for e_idx, e in enumerate(row.tolist())
                        if e is not None and e_idx < PP - 1]
            yield AllOf(env, procs)
            yield AllOf(env, pp_procs)

    yield env.process(schedule_procs(arr, lambda PP,pp: pp))

    xpu = xpus[PP - 1]
    tp_comm = tp_comms[0 * DP + PP - 1]
    yield env.process(loss_gradient(env, G, B, S, E, V, TP, dtype, tp_comm, dp_comm, xpu, dev_id))

    # Bkward pass
    arr = np.empty([PP, len(M) + PP - 1], dtype=object)
    for m_id, m in enumerate(M):
        for pp in range(PP):
            xpu = xpus[pp]
            arr[pp][m_id] = tformer_bk(pp, xpu, L=ll, B=m)

    # print("Params: ", xpus[0].param_count / GIGA)
    yield env.process(schedule_procs(arr, lambda PP, pp: PP-1-pp))
    flops = (xpus[0].flop_count)
    flop_max = xpus[0].flops[dtype.value - 1]
    print("flop_max:", flop_max)
    print("Comp time:", xpus[0].compute_time)
    print("TP Comm time:", tp_comms[0].comm_time)
    print("DP Comm time:", dp_comm.comm_time)
    if len(pp_comms) > 0:
        print("PP Comm time:", pp_comms[0].comm_time)
    print("train hours: ", num_iters, (flops * num_iters) / (60*60 * flop_max))
    print("FLOP diff:", flops/GIGA, (8 * param_count_per_dev * B * S) / GIGA)
