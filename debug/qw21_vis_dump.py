#!/usr/bin/env python3
"""
Qwen-Image 2.1 vision-tower stage dumper, for parity checks of the C code.

Loads only the text encoder's `model.visual.*` subtree (not the LM, not the
DiT), runs it on the pixels the editing pipeline would produce, and dumps
each stage as raw little-endian float32: interpolation indices/weights,
position ids, cu_seqlens, patch embed, interpolated pos embeds, rope cos/sin,
selected block outputs, the merged output and the three deepstack outputs.

Usage:
    ./flux_env/bin/python debug/qw21_vis_dump.py -d qwen-image-2.1/text_encoder \\
        -i ref.png --size 1024 -o /tmp/qw21_vis
"""

import argparse
import json
import os
import sys

import numpy as np

# torch/transformers/PIL/safetensors live in flux_env, not on a system
# interpreter; the try/except gives a usable message when run outside the venv.
try:
    # flux_env-only deps; Pyright resolves against the system interpreter.
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]
    from safetensors import safe_open  # type: ignore[import-not-found]
    from transformers import Qwen3VLProcessor  # type: ignore[import-not-found]
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (  # type: ignore[import-not-found]
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # type: ignore[import-not-found]
        Qwen3VLVisionModel,
        apply_rotary_pos_emb_vision,
    )
    from transformers.vision_utils import (  # type: ignore[import-not-found]
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )
except ImportError:
    sys.stderr.write(
        "Error: torch/transformers/PIL not found.\n"
        "Run with the project venv: ./flux_env/bin/python debug/qw21_vis_dump.py ...\n")
    sys.exit(1)


def dump_bin(out, name, tensor):
    arr = np.asarray(tensor.detach().float().cpu().numpy()
                     if hasattr(tensor, "detach") else tensor)
    path = os.path.join(out, name)
    try:
        with open(path, "wb") as f:
            f.write(arr.astype(np.float32).tobytes())
    except OSError as e:
        sys.exit(f"cannot write {path}: {e}")
    print(f"  {name}: shape={arr.shape}")


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError as e:
        sys.exit(f"cannot read {path}: {e}")


def load_vision(dir_path, cfg):
    vcfg = Qwen3VLVisionConfig(**cfg["vision_config"])
    dtype = torch.bfloat16 if cfg["vision_config"].get("dtype") == "bfloat16" else torch.float32
    vis = Qwen3VLVisionModel(vcfg).to(dtype)
    weight_map = load_json(os.path.join(dir_path, "model.safetensors.index.json"))["weight_map"]
    want = {k for k in weight_map if k.startswith("model.visual.")}
    loaded = 0
    for fname in sorted({weight_map[k] for k in want}):
        with safe_open(os.path.join(dir_path, fname), framework="pt") as g:
            keys = list(g.keys())
            state = {k[len("model.visual."):]: g.get_tensor(k)
                     for k in keys if k in want}
            vis.load_state_dict(state, strict=False)
            loaded += len(state)
    print("loaded vision tensors:", loaded)
    return vis


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dir", required=True, help="text_encoder dir")
    p.add_argument("--proc-dir", default=None,
                   help="dir with the processor configs (default: -d)")
    p.add_argument("-i", "--image", required=True)
    p.add_argument("--size", type=int, default=1024,
                   help="resize the image to size x size first")
    p.add_argument("-o", "--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        sys.exit(f"cannot create output dir {a.out}: {e}")

    try:
        img = Image.open(a.image)
    except OSError as e:
        sys.exit(f"cannot open {a.image}: {e}")
    if img.mode == "RGBA":  # the vision encoder sees alpha composited over white
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        img = white
    img = img.resize((a.size, a.size), Image.BICUBIC).convert("RGB")

    proc = Qwen3VLProcessor.from_pretrained(a.proc_dir or a.dir)
    inputs = proc(images=[img], return_tensors="pt")
    pixel_values = inputs.pixel_values
    grid = inputs.image_grid_thw
    print("grid_thw:", grid.tolist(), "pixel_values:", tuple(pixel_values.shape))

    cfg = load_json(os.path.join(a.dir, "config.json"))
    torch.set_grad_enabled(False)
    vis = load_vision(a.dir, cfg)

    vis_cfg = Qwen3VLVisionConfig(**cfg["vision_config"])
    gi, gw = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=int(vis_cfg.num_position_embeddings ** 0.5),
        mode="bilinear", align_corners=True,  # matches Qwen3VLVisionModel.interpolation_align_corners
        spatial_merge_size=vis_cfg.spatial_merge_size)
    pos_ids = get_vision_position_ids(grid, vis_cfg.spatial_merge_size)
    cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid, vis.config)
    dump_bin(a.out, "interp_idx.bin", gi)
    dump_bin(a.out, "interp_w.bin", gw)
    dump_bin(a.out, "pos_ids.bin", pos_ids)
    dump_bin(a.out, "cu_seqlens.bin", cu_seqlens)
    # Full (non-windowed) attention returns max_seqlen=None; compute it from the grid.
    if max_seqlen is None:
        max_seqlen = max(int(t) * int(h) * int(w) for t, h, w in grid.tolist())
    meta = {"grid": grid.tolist(), "pixel_shape": list(pixel_values.shape),
            "max_seqlen": int(max_seqlen),
            "weights_dtype": str(next(vis.parameters()).dtype)}
    try:
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump(meta, f)
    except OSError as e:
        sys.exit(f"cannot write meta.json: {e}")

    dump_bin(a.out, "pixel_values.bin", pixel_values)
    hs = vis.patch_embed(pixel_values)
    dump_bin(a.out, "after_patch_embed.bin", hs)
    pos_embeds = (vis.pos_embed.weight[gi] * gw[:, :, None]).sum(1)
    hs = hs + pos_embeds.to(hs.dtype)
    dump_bin(a.out, "pos_embeds.bin", pos_embeds)
    dump_bin(a.out, "after_pos.bin", hs)
    rope = vis.rotary_pos_emb(hs, pos_ids)
    dump_bin(a.out, "rope_cos.bin", rope[0])
    dump_bin(a.out, "rope_sin.bin", rope[1])

    # Block-0 internals for the C trace (q/k dumped heads-major, matching
    # the C per-head contiguous copies).
    b0 = vis.blocks[0]
    xn = b0.norm1(hs)
    dump_bin(a.out, "b0_ln1.bin", xn)
    heads, hd = vis_cfg.num_heads, vis_cfg.hidden_size // vis_cfg.num_heads
    seq = hs.shape[0]
    qkv = b0.attn.qkv(xn).reshape(seq, 3, heads, hd).permute(1, 0, 2, 3)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q, k = apply_rotary_pos_emb_vision(q, k, rope[0], rope[1])
    dump_bin(a.out, "b0_q.bin", q.transpose(0, 1))
    dump_bin(a.out, "b0_k.bin", k.transpose(0, 1))
    dump_bin(a.out, "b0_v.bin", v.transpose(0, 1))
    attn = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0))
    attn = attn.transpose(1, 2).reshape(seq, -1)
    dump_bin(a.out, "b0_attnout.bin", attn)
    proj = b0.attn.proj(attn)
    dump_bin(a.out, "b0_attnproj.bin", proj)
    x1 = hs + proj
    dump_bin(a.out, "b0_x1.bin", x1)
    xn2 = b0.norm2(x1)
    dump_bin(a.out, "b0_ln2.bin", xn2)
    f1 = b0.mlp.linear_fc1(xn2)
    dump_bin(a.out, "b0_fc1.bin", f1)
    act = b0.mlp.act_fn(f1)
    dump_bin(a.out, "b0_gelu.bin", act)
    dump_bin(a.out, "b0_fc2.bin", b0.mlp.linear_fc2(act))

    captured = {}
    probe_layers = (0, 8, 16, 24, 26)
    for ln in probe_layers:
        blk = vis.blocks[ln]
        orig = blk.forward

        def wrapped(*args, _orig=orig, _ln=ln, **kwargs):
            out = _orig(*args, **kwargs)
            captured[_ln] = out[0] if isinstance(out, tuple) else out
            return out
        blk.forward = wrapped

    hid = hs
    for blk in vis.blocks:
        hid = blk(hid, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                  position_embeddings=rope)
    for ln in probe_layers:
        dump_bin(a.out, f"block{ln:02d}.bin", captured[ln])
    dump_bin(a.out, "merged.bin", vis.merger(hid))
    for i, ln in enumerate(vis.deepstack_visual_indexes):
        dump_bin(a.out, f"deepstack{i}.bin", vis.deepstack_merger_list[i](captured[ln]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
