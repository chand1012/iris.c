#!/usr/bin/env python3
"""Dump Qwen-Image 2.1 vision-tower stage intermediates for C parity.

Loads only the text_encoder's visual submodel (not the LM or the DiT), runs
it on the same preprocessed pixels the editing pipeline produces, and dumps
each stage: interpolation indices/weights, position ids, patch embed, pos
embeds, rope cos/sin, selected block outputs, and the merged/deepstack outputs.
"""
import argparse
import json
import os
import sys

# torch/PIL/safetensors/transformers live in flux_env, not on a system
# interpreter; the try/except gives a usable message outside the venv.
try:
    # flux_env-only deps; Pyright resolves against the system interpreter.
    import numpy as np  # type: ignore[import-not-found]
    import torch  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]
    from safetensors import safe_open  # type: ignore[import-not-found]
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (  # type: ignore[import-not-found]
        Qwen3VLVisionModel,
    )
except ImportError:
    sys.stderr.write(
        "Error: torch/PIL/safetensors/transformers not found.\n"
        "Run with the project venv: ./flux_env/bin/python debug/qw21_vis_dump.py ...\n")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dir", required=True, help="text_encoder dir")
    p.add_argument("-i", "--image", required=True)
    p.add_argument("--size", type=int, default=1024,
                   help="resize the image to size x size first")
    p.add_argument("-o", "--out", required=True)
    return p.parse_args()


def dump(out, name, t):
    a = np.asarray(t.detach().float().cpu().numpy() if hasattr(t, "detach") else t)
    a.tofile(os.path.join(out, name))
    print(f"  {name}: shape={a.shape}")


def main():
    a = parse_args()
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        sys.exit(f"cannot create output dir {a.out}: {e}")

    # flux_env-only deps; Pyright resolves against the system interpreter.
    from transformers import Qwen3VLProcessor  # type: ignore[import-not-found]
    from transformers.vision_utils import (  # type: ignore[import-not-found]
        get_vision_attention_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    img = Image.open(a.image)
    if img.mode == "RGBA":  # vision encoder sees alpha composited over white
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        img = white
    img = img.resize((a.size, a.size), Image.BICUBIC).convert("RGB")

    proc = Qwen3VLProcessor.from_pretrained(a.dir)
    inputs = proc(images=[img], return_tensors="pt")
    pixel_values = inputs.pixel_values
    grid = inputs.image_grid_thw
    print("grid_thw:", grid.tolist(), "pixel_values:", tuple(pixel_values.shape))

    try:
        with open(os.path.join(a.dir, "config.json")) as f:
            cfg = json.load(f)
    except OSError as e:
        sys.exit(f"cannot read config: {e}")
    vis_cfg = cfg["vision_config"]
    torch.set_grad_enabled(False)
    dtype = torch.bfloat16 if vis_cfg.get("dtype") == "bfloat16" else torch.float32
    vis = Qwen3VLVisionModel(vis_cfg).to(dtype)
    idx_path = os.path.join(a.dir, "model.safetensors.index.json")
    try:
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
    except OSError as e:
        sys.exit(f"cannot read {idx_path}: {e}")
    want = {k for k in weight_map if k.startswith("model.visual.")}
    loaded = 0
    for fname in sorted({weight_map[k] for k in want}):
        with safe_open(os.path.join(a.dir, fname), framework="pt") as g:
            state = {k[len("model.visual."):]: g.get_tensor(k)
                     for k in list(g.keys()) if k in want}
            vis.load_state_dict(state, strict=False)
            loaded += len(state)
    print("loaded vision tensors:", loaded)

    gi, gw = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=int(vis_cfg["num_position_embeddings"] ** 0.5),
        mode="bilinear", align_corners=True,
        spatial_merge_size=vis_cfg["spatial_merge_size"])
    pos_ids = get_vision_position_ids(grid, vis_cfg["spatial_merge_size"])
    cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid, vis.config)
    dump(a.out, "interp_idx.bin", gi)
    dump(a.out, "interp_w.bin", gw)
    dump(a.out, "pos_ids.bin", pos_ids)
    dump(a.out, "cu_seqlens.bin", cu_seqlens)
    try:
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump({"grid": grid.tolist(), "pixel_shape": list(pixel_values.shape),
                       "weights_dtype": str(next(vis.parameters()).dtype)}, f)
    except OSError as e:
        sys.exit(f"cannot write meta.json: {e}")

    dump(a.out, "pixel_values.bin", pixel_values)
    hs = vis.patch_embed(pixel_values)
    dump(a.out, "after_patch_embed.bin", hs)
    pos_embeds = (vis.pos_embed.weight[gi] * gw[:, :, None]).sum(1)
    hs = hs + pos_embeds.to(hs.dtype)
    dump(a.out, "pos_embeds.bin", pos_embeds)
    dump(a.out, "after_pos.bin", hs)
    rope = vis.rotary_pos_emb(hs, pos_ids)
    dump(a.out, "rope_cos.bin", rope[0])
    dump(a.out, "rope_sin.bin", rope[1])

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
        dump(a.out, f"block{ln:02d}.bin", captured[ln])
    dump(a.out, "merged.bin", vis.merger(hid))
    for i, ln in enumerate(vis.deepstack_visual_indexes):
        dump(a.out, f"deepstack{i}.bin", vis.deepstack_merger_list[i](captured[ln]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
