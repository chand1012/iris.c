#!/usr/bin/env python3
"""Dump Qwen-Image 2.1 editing-path transformer intermediates.

Runs the pipeline with one condition image and one denoising step, capturing:
  - the raw joint latents fed to img_in  (refs + target)
  - encoder_hidden_states fed to txt_in
  - the assembled joint sequence after repeat_interleave + latent scatter
  - rope freqs, image_ids, target_token_mask, key_valid
Used to debug the C editing forward against the reference.
"""
import argparse
import gc
import json
import os
import sys
import types

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dir", required=True)
    p.add_argument("-p", "--prompt", required=True)
    p.add_argument("-i", "--ref", action="append", required=True)
    p.add_argument("-W", "--width", type=int, default=256)
    p.add_argument("-H", "--height", type=int, default=256)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("-o", "--out", required=True)
    return p.parse_args()


def dump(out, name, t) -> np.ndarray:
    a = t.detach().float().cpu().numpy()
    a.tofile(os.path.join(out, name))
    print(f"  {name}: {a.shape} {a.dtype}")


def main():
    a = parse_args()
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        sys.exit(f"cannot create output dir {a.out}: {e}")

    from PIL import Image
    from diffusers import QwenImage21Pipeline
    print(f"Loading {a.dir} (float32, CPU) ...")
    pipe = QwenImage21Pipeline.from_pretrained(
        a.dir, torch_dtype=torch.float32, device_map="cpu")

    state = {}
    orig_encode = pipe.encode_prompt

    def hook_encode(*args, **kw):
        out = orig_encode(*args, **kw)
        state["embeds"] = out[0]
        state["enc_mask"] = out[1]
        state["img_mask"] = out[2]
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder = None
            gc.collect()
        return out
    pipe.encode_prompt = hook_encode

    tf = pipe.transformer
    orig_img_in = tf.img_in.forward

    def hook_img_in(self, hs):
        state["img_in_x"] = hs
        return orig_img_in(hs)
    tf.img_in.forward = types.MethodType(hook_img_in, tf.img_in)

    orig_txt_in = tf.txt_in.forward

    def hook_txt_in(self, hs):
        state["txt_in_x"] = hs
        return orig_txt_in(hs)
    tf.txt_in.forward = types.MethodType(hook_txt_in, tf.txt_in)

    orig_forward = tf.forward

    def hook_forward(self, *args, **kw):
        # Re-create the internals exactly as the model does, once.
        hs = kw["hidden_states"]
        ehs = kw["encoder_hidden_states"]
        img_mask = kw["img_mask"]
        shapes = kw["img_shapes"][0]
        with torch.no_grad():
            iin = tf.img_in(hs)
            tin = tf.txt_in(ehs)
            repeats = torch.where(img_mask, 4, 1)[0]
            pad_mask = torch.repeat_interleave(img_mask[0], repeats)
            import math
            target_tokens = math.prod(shapes[-1])
            joint = torch.cat([tin, tin.new_zeros(1, target_tokens // 4, tin.shape[2])], dim=1)
            joint = joint.repeat_interleave(repeats, dim=1)
            joint[:, pad_mask] = iin
            rope = tf.pos_embed(shapes, pad_mask, device=hs.device)
            image_ids, target_token_mask = tf.build_token_metadata(pad_mask, shapes)
        state["joint"] = joint
        state["rope"] = rope
        state["image_ids"] = image_ids
        state["target_mask"] = target_token_mask
        # skip the real forward for step 1 only? keep it for velocity
        return orig_forward(*args, **kw)
    tf.forward = types.MethodType(hook_forward, tf)

    refs = [Image.open(p) for p in a.ref]
    gen = torch.Generator("cpu").manual_seed(a.seed)
    out = pipe(
        prompt=a.prompt, image=refs, width=a.width, height=a.height,
        num_inference_steps=a.steps, generator=gen, use_kv_cache=False,
        output_type="pil")
    out.images[0].save(os.path.join(a.out, "image.png"))

    print("Dumps:")
    dump(a.out, "img_in_x.bin", state["img_in_x"])          # [1, n_img, 64]
    dump(a.out, "txt_in_x.bin", state["txt_in_x"])          # [1, n_slots_vlm, 4096]
    dump(a.out, "joint.bin", state["joint"])                # [1, N, 4096]
    r = state["rope"]
    dump(a.out, "rope_mag.bin", r.abs())
    dump(a.out, "rope_pha.bin", r.angle())
    try:
        with open(os.path.join(a.out, "meta.json"), "w") as f:
            json.dump({"N": int(state["joint"].shape[1]),
                       "n_img": int(state["img_in_x"].shape[1]),
                       "n_vlm": int(state["txt_in_x"].shape[1])}, f)
    except OSError as e:
        sys.exit(f"cannot write meta.json: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
