#!/usr/bin/env python3
"""Dump Qwen-Image 2.1 transformer intermediates for C-parity bisection.

Runs the diffusers QwenImage21Transformer2DModel (float32, CPU) on the exact
inputs a parity dump was made from (noise.bin / emb.bin / img_mask_full.bin /
img_shapes.json), checks that it reproduces step0_v.bin, and writes every
captured intermediate as a float32 .bin plus a manifest.json.

Usage: flux_env/bin/python debug/qw21_mid_dump.py <model_dir> <dump_dir> [--out DIR]
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("dump_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or a.dump_dir.rstrip("/") + "_mids"
    os.makedirs(out, exist_ok=True)

    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21Transformer2DModel,
    )

    tf = QwenImage21Transformer2DModel.from_pretrained(
        os.path.join(a.model_dir, "transformer"), torch_dtype=torch.float32
    )
    tf.eval()
    print("config:", dict(tf.config))

    noise = np.fromfile(os.path.join(a.dump_dir, "noise.bin"), np.float32)
    emb = np.fromfile(os.path.join(a.dump_dir, "emb.bin"), np.float32)
    mask = np.fromfile(os.path.join(a.dump_dir, "img_mask_full.bin"), np.float32)
    shapes = json.load(open(os.path.join(a.dump_dir, "img_shapes.json")))
    refv = np.fromfile(os.path.join(a.dump_dir, "step0_v.bin"), np.float32)

    n_target = refv.size // 64
    n_txt = emb.size // 4096
    in_ch = tf.config.in_channels
    ctx = tf.config.context_in_dim
    print(f"n_txt={n_txt} n_target={n_target} mask_len={mask.size} in_ch={in_ch} ctx={ctx}")

    hs = torch.from_numpy(noise).view(1, n_target, in_ch)
    ehs = torch.from_numpy(emb).view(1, n_txt, ctx)
    imask = torch.from_numpy(mask).view(1, -1).bool()
    t = torch.tensor([1.0])  # step-0 sigma (pipeline feeds raw sigma)

    mid = {}

    def cap(name):
        def hook(mod, inp, outp):
            o = outp[0] if isinstance(outp, (tuple, list)) else outp
            if isinstance(o, torch.Tensor):
                mid[name] = o.detach().float().cpu().numpy()
        return hook

    def cap_in(name):
        def hook(mod, inp):
            x = inp[0]
            if isinstance(x, torch.Tensor):
                mid[name] = x.detach().float().cpu().numpy()
        return hook

    def reg(name, mod, kind="out"):
        if mod is None or not isinstance(mod, torch.nn.Module):
            print(f"MISS {name} -> {type(mod).__name__ if mod is not None else None}")
            return
        if kind == "in":
            mod.register_forward_pre_hook(cap_in(name))
        else:
            mod.register_forward_hook(cap(name))

    for n, m in tf.named_children():
        print("TOP", n, type(m).__name__)
    b0 = tf.transformer_blocks[0]
    for n, m in b0.named_children():
        print("B0", n, type(m).__name__)
    for n, m in tf.txt_in.named_children():
        print("TXTIN", n, type(m).__name__)
    print("B0.ATTN", [n for n, _ in b0.attn.named_children()])
    print("B0.MLP", [n for n, _ in b0.img_mlp.named_children()])
    print("NORMOUT", [n for n, _ in tf.norm_out.named_children()] if hasattr(tf, "norm_out") else "n/a")

    reg("time_proj", getattr(getattr(tf, "time_text_embed", None), "time_proj", None))
    reg("timestep_embedder", getattr(getattr(tf, "time_text_embed", None), "timestep_embedder", None))
    reg("time_text_embed", tf.time_text_embed)
    reg("modulation", getattr(tf, "modulation", None))
    reg("img_in", getattr(tf, "img_in", None))
    reg("txt_in", getattr(tf, "txt_in", None))
    reg("txt_norm", getattr(tf.txt_in, "text_norm", None))
    reg("txt_in_layer", getattr(tf.txt_in, "in_layer", None))
    reg("txt_out_layer", getattr(tf.txt_in, "out_layer", None))
    reg("b0.out", b0)
    reg("b0.img_norm1", b0.img_norm1)
    reg("b0.xm", b0.attn.to_q, "in")
    reg("b0.to_q", b0.attn.to_q)
    reg("b0.to_k", b0.attn.to_k)
    reg("b0.to_v", b0.attn.to_v)
    reg("b0.norm_q", b0.attn.norm_q)
    reg("b0.norm_k", b0.attn.norm_k)
    reg("b0.attn_out", b0.attn)
    reg("b0.to_out_in", b0.attn.to_out[0], "in")
    reg("b0.xm2", b0.img_mlp.gate_layer, "in")
    reg("b0.mlp_gate", b0.img_mlp.gate_layer)
    reg("b0.mlp_up", b0.img_mlp.proj)
    reg("b0.mlp_out", b0.img_mlp.out)
    reg("b31.out", tf.transformer_blocks[-1])
    reg("norm_out", getattr(tf, "norm_out", None))
    reg("proj_out", getattr(tf, "proj_out", None))

    with torch.no_grad():
        o = tf(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            timestep=t,
            img_shapes=[shapes],
            img_mask=imask,
            return_dict=False,
        )[0]
    got = o[0, -n_target:].cpu().numpy()
    d = np.abs(got.reshape(-1) - refv)
    print(f"SANITY model vs step0_v: max={d.max():.6g} mean={d.mean():.6g}")

    man = {}
    for name, arr in sorted(mid.items()):
        p = os.path.join(out, name + ".bin")
        arr.astype(np.float32).tofile(p)
        man[name] = list(arr.shape)
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    print("captured:", json.dumps(man))


if __name__ == "__main__":
    main()
