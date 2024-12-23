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

def loss_gradient(env, G, B, S, E, V, TP, dtype, tp_comm, dp_comm, xpu, dev_id):
    yield env.process(xpu.matmul(1, B*S, E, (V // TP), dtype, True, [f"X@vocab"]))
    yield env.process(tp_comm.all_gather(B*S*V, dtype, [f"xpu{dev_id}-vocab_out-gather"]))

    yield env.process(dp_comm.all_reduce(G*S*V, dtype, [f"xpu_{dev_id}_loss_grad_partial"]))
    yield env.process(xpu.matmul_bk(1, B*S, E, (V // TP), 1, dtype, [f"X@vocab"]))
    yield env.process(tp_comm.all_gather(B*S*V, dtype, [f"xpu{dev_id}-bk_vocab_out-gather"]))

def load_qkv_ffns(env, L, E, E_tp, H_tp, num_heads, kv_heads, is_llama_ffn, freeze, is_train, dtype, xpu, op):
    def add_opt_states(p, i, l):
        return [(xpu.mem_fill(p, Dtypes.FP32, op+[f"{k}_{i}"]))
                for k in ["momentum", "variance", "param"]]
    qkvs = []
    ffns = []
    for l in range(L):
        qkvs = qkvs + [(xpu.mem_fill(E*E_tp, dtype, op+[f"W_{i}"])) for i in ["Q"]]
        qkvs = qkvs + [(xpu.mem_fill(int(E*E_tp * kv_heads/num_heads), dtype, op+[f"W_{i}"])) for i in ["K", "V"]]
        FFNs = ["ffn1", "ffn2", "ffn3"] if is_llama_ffn else ["ffn1", "ffn2"]
        ffns = ffns + [(xpu.mem_fill(E*H_tp, dtype, op+[f"W_{i}"])) for i in FFNs]
        if is_train and freeze is False:
            os = [p for i in ["Q"] for p in add_opt_states(E*E_tp, i, l)]
            qkvs = qkvs + os
            os = [p for i in ["K", "V"] for p in add_opt_states(int(E*E_tp*kv_heads/num_heads), i, l)]
            os = [p for i in FFNs for p in add_opt_states(E*H_tp, i, l)]
            ffns = ffns + os
    return qkvs + ffns

def embed_fwd(env, B, S, E, V, E_tp, dtype, freeze, dev_id, tp_comm, xpu):
    yield env.process(xpu.mem_fill(B*S*E, dtype, ["X"]))
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
    attn_params = 2*E*E + 2*E*E*(kv_heads/num_heads)
    mlp_params = 3 * E * H if is_llama_mlp else 2 * E * H
    out_params = 2 * E * V
    param_count = attn_params + 4*E + mlp_params + E + H if has_bias else attn_params + mlp_params
    total_params = (param_count * L) + out_params
    print("PP layers:", layer_splits(PP, L))
    print("Total params:", total_params/GIGA)
    HB = cluster_specs.HB
    # tp comms
    tp_comms = [Ccl(env, [HB, TP//HB], bws, bw_eff) for i in range(DP)]
    dp_comm = Ccl(env, [TP, DP], bws, bw_eff)
    ll = L // PP
    b = B // PP
    print("p, b", p, b)
    # Only ever has one destination and one sender
    pp_comms = [Ccl(env, [1, 1], bws, bw_eff) for i in range(PP)]
    hps = Hps(env, hps_rd_bw=1000*GIGA, hps_wr_bw=500*GIGA)
    print("Before tformer CID:", get_cid())
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
    def tformer(dev_id, L=1):
        print("Inside tformer CID:", get_cid())
        tp_comm = tp_comms[dev_id // TP]
        #pp_comm = pp_comms[pp_group_id]
        def add_opt_states(p, i, l):
            return [env.process(xpu.mem_fill(p, Dtypes.FP32, [f"{k}_{i}"]))
                    for k in ["momentum", "variance", "param"]]

        qkvs = []
        ffns = []
        xpu = Xpu(env, xpu_specs, dev_id) # Every xpu takes up two cids
        # Weight load
        if is_text:
            yield env.process(xpu.mem_fill((V//TP)*E, dtype, ["embed_load"]))
        elif is_vision:
            yield env.process(xpu.mem_fill(E_tp * vision_specs.C * vision_specs.P * vision_specs.P, dtype, ["conv_kernel_load"]))

        qkv_ffns = load_qkv_ffns(env, L, E, E_tp, H_tp, num_heads, kv_heads,
                                 is_llama_mlp, freeze, is_train, dtype, xpu, [])
        yield AllOf(env, [env.process(l) for l in qkv_ffns])
        if is_text:
            yield env.process(xpu.mem_fill(E * (V // TP), dtype, ["vocab_load"]))
            yield env.process(embed_fwd(env, B, S, E, V, E_tp, dtype, freeze, dev_id, tp_comm, xpu))
        elif is_vision:
            yield env.process(img_emb_fwd(env, B, H_out, W_out, C, K, E, E_tp, dtype, freeze, dev_id, tp_comm, xpu))

        # Train
        for l in range(L):
            # if dev_id == 0:
            #     print(f"Training layer-{l}")
            yield env.process(fwd_pass(env, B, S, E, V, E_tp, H_tp, dtype,
                                       num_heads, kv_heads, is_llama_mlp, freeze,
                                       dev_id, tp_comm, xpu, ckpt=False, op=[]))

# loss_gradient(env, G, B, S, E, V, TP, dtype, tp_comm, dp_comm, xpu, dev_id):
        yield env.process(loss_gradient(env, G, B, S, E, V, TP, dtype, tp_comm, dp_comm, xpu, dev_id))
        for l in range(L):
            yield env.process(bk_pass(env, B, S, E, V, E_tp, H_tp, dtype,
                                      num_heads, kv_heads, is_llama_mlp, freeze, dev_id,
                                      tp_comm, xpu, op=[]))

        if is_text and (not freeze):
            # all-scatter
            yield env.process(tp_comm.all_gather(B*S*E, dtype, [f"xpu{dev_id}-bk-embed-gather"]))
            yield env.process(xpu.matmul_bk(1, B*S, (V//TP), E, 1, dtype, [f"X@emb"]))
        if is_vision and (not freeze):
            yield env.process(tp_comm.all_gather(B*H_out*W_out*E, dtype,[f"img-emb-gather"]))
            yield env.process(xpu.matmul_bk(1, B*H_out*W_out, C*K*K, E_tp, 1, dtype, ["image-embed-gen"]))
        yield env.process(hps.write(param_count / (DP * TP), dtype, [f"xpu{dev_id}_wt_ckpt"]))
        yield env.process(hps.write(3 * param_count / (DP * TP), Dtypes.FP32, [f"xpu{dev_id}_opt_ckpt"]))

        if dev_id == 0:
            print("Free mem:", xpu.mem_rem())
        #     for k, v in xpu.mem_contents.items():
        #         if v > 0:
        #             print(k, v)
    # For every batch in m
    # go through fwd pass
    # initiate pp_comm
    # wait_until_done
    # reverse layers
    for pp in range(PP):
        yield env.process(tformer(pp, L=ll))
    # yield AllOf(env, [start_delayed(env, tformer(i, L=ll), i+1) for i in range(1)])
