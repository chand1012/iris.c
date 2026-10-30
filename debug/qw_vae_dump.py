#!/usr/bin/env python3
"""Dump Qwen-Image 2.1 VAE intermediates (official _encode/_decode path)
on a deterministic 32x32 RGBA image, for C parity checks.

Outputs (raw float32 .npy, frame 0 of any T dim):
  00_image      [4,32,32]      input image in [-1,1]
  13_enc_quant  [128,2,2]      quant_conv moments (mean | logvar)
  14_enc_mean   [64,2,2]       latent mean
  15_latent     [64,2,2]       normalized latent: (mean - lmean)/lstd
  90_dec_full   [4,32,32]      vae._decode(latents) output, clamped [-1,1]
"""
import os
import shutil
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "debug", "qw_vae_dumps")
if os.path.isdir(OUT):
    shutil.rmtree(OUT)
os.makedirs(OUT)


def gen_image(h, w, seed=0):
    st = np.uint32(seed)

    def nxt():
        nonlocal st
        st = (st * 1664525 + 1013904223) & np.uint32(0xFFFFFFFF)
        return int(st >> 8)

    a = np.empty((4, h, w), np.float32)
    for c in range(4):
        for y in range(h):
            for x in range(w):
                a[c, y, x] = (nxt() & 0xFF) / 127.5 - 1.0
    return a


def dump(name, t):
    t = t.detach().float().cpu()
    if t.dim() == 5:      # (B, C, T, H, W) -> frame 0
        t = t[0, :, 0]
    elif t.dim() == 4:    # (B, C, H, W)
        t = t[0]
    arr = t.numpy()
    np.save(os.path.join(OUT, name + ".npy"), arr)
    print(f"{name}: {list(arr.shape)}", flush=True)


def main():
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import (
        AutoencoderKLQwenImage21,
    )

    vae = AutoencoderKLQwenImage21.from_pretrained(
        os.path.join(ROOT, "qwen-image-2.1", "vae"), torch_dtype=torch.float32
    ).eval()

    size = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    x5 = torch.tensor(gen_image(size, size)).unsqueeze(0).unsqueeze(2).to(vae.device)
    dump("00_image", x5)

    if os.environ.get("QW_VAE_HOOKS"):
        enc, dec = vae.encoder, vae.decoder
        mods = [("p_enc_conv_in", enc.conv_in), ("p_enc_mid", enc.mid_block),
                ("p_enc_convout", enc.conv_out), ("p_dec_convin", dec.conv_in),
                ("p_dec_mid", dec.mid_block), ("p_dec_convout", dec.conv_out)]
        mods += [(f"p_enc_b{i}", enc.down_blocks[i]) for i in range(5)]
        b1 = enc.down_blocks[1]
        mods += [("p_b1_r0", b1.resnets[0]), ("p_b1_r1", b1.resnets[1]),
                ("p_b1_dn", b1.downsampler), ("p_b1_sh", b1.avg_shortcut)]
        mb = enc.mid_block
        mods += [("p_mid_a", mb.resnets[0]), ("p_mid_attn", mb.attentions[0]),
                ("p_mid_b", mb.resnets[1])]
        at = mb.attentions[0]
        mods += [("p_a_norm", at.norm), ("p_a_qkv", at.to_qkv)]
        mods += [(f"p_dec_u{i}", dec.up_blocks[i]) for i in range(5)]
        for name, m in mods:
            m.register_forward_hook(lambda mo, i, o, _n=name: dump(_n, o))

    torch.set_grad_enabled(False)
    moments = vae._encode(x5)
    dump("13_enc_quant", moments)
    mean = moments[:, :64]
    dump("14_enc_mean", mean)

    lm = torch.tensor(vae.config.latents_mean, device=mean.device).view(1, 64, 1, 1, 1)
    ls = torch.tensor(vae.config.latents_std, device=mean.device).view(1, 64, 1, 1, 1)
    lat = (mean - lm) / ls
    dump("15_latent", lat)

    if os.environ.get("QW_VAE_HOOKS"):
        ls5 = lat * ls + lm
        dump("p_dec_denorm", ls5)
        dump("p_dec_postq", vae.post_quant_conv(ls5))

    out = vae._decode(lat * ls + lm).sample
    dump("90_dec_full", out)
    print("decode T:", out.shape[2], flush=True)
    torch.set_grad_enabled(True)


if __name__ == "__main__":
    main()
