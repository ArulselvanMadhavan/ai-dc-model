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
from models.tformer import embed_fwd, fwd_pass, load_qkv_ffns, img_emb_fwd, loss_gradient, bk_pass

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
    tp_comm = tp_comms[dev_id // DP]
    vis_model_freq = 4
    assert txt_model_specs.L % vis_model_freq == 0
    L_xattn = txt_model_specs.L // vis_model_freq
    def im_model_run(model_specs):
        vision_specs = model_specs.vision
        H_out = calc_conv_out_dim(vision_specs.H, 0, 1, vision_specs.P, vision_specs.P)
        W_out = calc_conv_out_dim(vision_specs.W, 0, 1, vision_specs.P, vision_specs.P)
        S = H_out * W_out
        C = vision_specs.C
        K = vision_specs.P
        G = model_specs.G
        B = G // DP
        E = model_specs.E
        H = model_specs.H
        H_tp = H // TP
        E_tp = model_specs.E // TP
        V = model_specs.V
        L = model_specs.L
        dtype = model_specs.param_dtype
        freeze = model_specs.freeze
        num_heads = model_specs.num_heads
        kv_heads = model_specs.kv_heads
        is_llama_mlp = False
        LF = 6
        param_count = 4*E*E + 4*E + 2*E*H + E + H if has_bias else 4*E*E + 2*E*H
        assert E % TP == 0, "TP dim size should divide embedding dimension"
        freeze = model_specs.freeze
        print("im-model params:", (param_count * L + (6*E*8192))/GIGA)
        kload = xpu.mem_fill(E_tp * vision_specs.C * vision_specs.P * vision_specs.P, dtype, ["conv_kernel_load"])
        qkv_ffns = load_qkv_ffns(env, L, E, E_tp, H_tp, num_heads, kv_heads,
                                 False, freeze, is_train, dtype, xpu, ["im"])
        out_proj = xpu.mem_fill(LF*E_tp*LF*E_tp, dtype, ["im-out-proj"])
        yield [kload] + qkv_ffns + [out_proj]
        def img_fwd():
            yield env.process(img_emb_fwd(env, B, H_out, W_out, C, K, E, E_tp, dtype, False, dev_id, tp_comm, xpu))
            for i in range(L):
                yield env.process(fwd_pass(env, B, S, E, V, E_tp, H_tp, dtype,
                                           num_heads, kv_heads, is_llama_mlp, freeze, dev_id,
                                           tp_comm, xpu, ckpt=False, op=["im"]))
            yield env.process(xpu.matmul(1, B*S, LF*E_tp,
                                         txt_model_specs.E // TP,
                                         dtype, True, [f"im-multilayer-proj"]))
            yield env.process(tp_comm.all_reduce(B*S*LF*E, dtype, [f"xpu{dev_id}-im-partial"]))
        def img_bk():
            yield env.process(xpu.matmul_bk(1, B*S, LF*E_tp,
                                         txt_model_specs.E // TP,
                                         dtype, [f"im-multilayer-proj"]))
            for i in range(L):
                yield env.process(bk_pass(env, B, S, E, V, E_tp, H_tp, dtype, num_heads, kv_heads,
                                          is_llama_mlp, freeze, dev_id, tp_comm, xpu, ckpt=False, op=["im"]))
        for i in range(L_xattn):
            yield img_fwd()
        for i in range(L_xattn):
            yield img_bk()


    def txt_model_run(model_specs):
        G = model_specs.G
        B = G // DP
        S = model_specs.S
        E = model_specs.E
        assert E % TP == 0, "TP dim size should divide embedding dimension"
        H = model_specs.H
        H_tp = H // TP
        E_tp = model_specs.E // TP
        V = model_specs.V
        L = model_specs.L
        num_heads = model_specs.num_heads
        kv_heads = model_specs.kv_heads
        is_llama_mlp = True
        bws = [cluster_specs.scale_up, cluster_specs.scale_out]
        bw_eff = [0.7, 0.7]
        dtype = Dtypes.FP16
        is_train = model_specs.is_train
        has_bias = False
        param_count = 4*E*E + 4*E + 3*E*H + E + H if has_bias else 2*E*E + 2*E*E*(kv_heads/num_heads) + 3*E*H
        out_params = 2 * E * V
        freeze = model_specs.freeze
        print("txt-model-params:", ((param_count * L) + out_params)/GIGA)
        eload = xpu.mem_fill(V*E_tp, dtype, ["embed_load"])
        qkv_ffns = load_qkv_ffns(env, L, E, E_tp, H_tp, num_heads, kv_heads,
                                 True, freeze, is_train, dtype, xpu, ["txt"])
        loads = [eload] + qkv_ffns
        vocab_load = xpu.mem_fill(E * (V // TP), dtype, ["vocab_load"])
        yield loads + [vocab_load]
        yield embed_fwd(env, B, S, E, V, E_tp, dtype, freeze, dev_id, tp_comm, xpu)
        for l in range(1, L+1):
            if l % vis_model_freq == 0:
                yield fwd_pass(env, B, S, E, V, E_tp, H_tp, dtype,
                               num_heads, kv_heads, True, freeze, dev_id,
                               tp_comm, xpu, ckpt=False, op=["xattn"])
            else:
                yield fwd_pass(env, B, S, E, V, E_tp, H_tp, dtype,
                               num_heads, kv_heads, True, freeze,
                               dev_id, tp_comm, xpu, ckpt=False, op=["txt"])
        yield loss_gradient(env, G, B, S, E, V, TP, dtype, tp_comm, dp_comm, xpu, dev_id)
        for l in range(1, L+1):
            if l % vis_model_freq == 0:
                yield bk_pass(env, B, S, E, V, E_tp, H_tp, dtype, num_heads, kv_heads,
                              is_llama_mlp, freeze, dev_id, tp_comm, xpu, ckpt, op=["xattn"])
            else:
                yield bk_pass(env, B, S, E, V, E_tp, H_tp, dtype, num_heads, kv_heads,
                              is_llama_mlp, freeze, dev_id, tp_comm, xpu, ckpt, op=["txt"])

    im_model_gen = im_model_run(im_model_specs)
    txt_model_gen = txt_model_run(txt_model_specs)
    im_model_load = next(im_model_gen)
    txt_model_load = next(txt_model_gen)
    all_loads = im_model_load + txt_model_load
    chunk_size = 10
    chunk = []
    for i, l in enumerate(all_loads):
        chunk.append(l)
        if i % chunk_size == 0:
            yield AllOf(env, [env.process(c) for c in chunk])
            chunk = []
    yield AllOf(env, [env.process(c) for c in chunk])
    yield env.process(next(txt_model_gen)) # embed_fwd
    for l in range(1, txt_model_specs.L+1):
        if l % vis_model_freq == 0:
            yield env.process(next(im_model_gen))
        yield env.process(next(txt_model_gen))

    yield env.process(next(txt_model_gen))

    for l in range(1, txt_model_specs.L+1):
        if l % vis_model_freq == 0:
            yield env.process(next(im_model_gen))
        yield env.process(next(txt_model_gen))

    print(xpu.mem_rem())
    # yield AllOf(env, [env.process(ll) for ll in im_model_load])
    # print(xpu.mem_rem())
    # yield AllOf(env, all_loads)
