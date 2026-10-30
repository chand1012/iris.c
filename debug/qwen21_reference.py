#!/usr/bin/env python3
"""
Qwen-Image 2.1 reference pipeline, for parity checks of the C code.

Runs the official diffusers QwenImage21Pipeline (0.42+; the 2.1 pipeline is
in flux_env) in float32 on CPU and dumps every intermediate tensor as raw
little-endian float32 so C test harnesses can compare against them.

Usage:
    ./flux_env/bin/python debug/qwen21_reference.py -d qwen-image-2.1 \\
        -p "a fish" -W 512 -H 512 --steps 4 --seed 42 -o /tmp/qwen21_ref
    # with reference images (editing path):
    ./flux_env/bin/python debug/qwen21_reference.py -d qwen-image-2.1 \\
        -p "oil painting" -i cat.png -W 1024 -H 1024 -o /tmp/qwen21_ref

Outputs in the -o directory:
    tokens.txt            processor input ids (padded), one per line
    drop_idx.txt          leading system tokens dropped from hidden states
    emb.bin               prompt embeddings [txt_seq, 4096] (post-drop, right-padded)
    enc_mask.bin          encoder attention mask [txt_seq]
    img_mask.bin          image-slot mask, text stream only [txt_seq]
    img_mask_full.bin     mask with target-image slots appended (transformer input)
    img_shapes.json       per-block (frame, h, w) in latent tokens, target last
    ref<i>.bin            reference i's VAE latents, packed+normalized [h*w, 64]
    noise.bin             initial target latents [h*w, 64]
    sigmas.txt            final sigma schedule (num_steps + 1 values)
    step<i>_v.bin         velocity predicted at step i, target tokens [h*w, 64]
    step<i>_x.bin         target latents after step i [h*w, 64]
    latent.bin            final latents before VAE denorm [h*w, 64]
    image.png             decoded RGBA image

Pass --noise FILE to start from a noise file (e.g. one dumped by C).
--kv-cache enables the pipeline's prefix KV cache (off by default: the C
transformer recomputes the full sequence every step in the first phase).
"""

import argparse
import gc
import json
import os
import sys
import tempfile

import numpy as np

# torch/diffusers/PIL live in flux_env, not on a system interpreter; the
# try/except gives a usable message when the script is run outside the venv.
try:
    # flux_env-only deps; Pyright resolves against the system interpreter.
    import torch  # type: ignore[import-not-found]
    from diffusers import QwenImage21Pipeline  # type: ignore[import-not-found]
    from PIL import Image  # type: ignore[import-not-found]
except ImportError:
    sys.stderr.write(
        "Error: torch/diffusers/PIL not found.\n"
        "Run with the project venv: ./flux_env/bin/python debug/qwen21_reference.py ...\n")
    sys.exit(1)


def dump_bin(out_dir, name, tensor):
    arr = tensor.detach().float().cpu().numpy().astype(np.float32)
    path = os.path.join(out_dir, name)
    try:
        with open(path, "wb") as f:
            f.write(arr.tobytes())
    except OSError as e:
        sys.exit(f"cannot write {path}: {e}")


def write_text(out_dir, name, text):
    path = os.path.join(out_dir, name)
    try:
        with open(path, "w") as f:
            f.write(text)
    except OSError as e:
        sys.exit(f"cannot write {path}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-d", "--dir", required=True, help="model directory")
    p.add_argument("-p", "--prompt", required=True)
    p.add_argument("-W", "--width", type=int, default=512)
    p.add_argument("-H", "--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-i", "--ref", action="append", default=[],
                   help="reference image (repeatable)")
    p.add_argument("--noise", default=None, help="start from this packed noise file")
    p.add_argument("--kv-cache", action="store_true",
                   help="enable the pipeline's prefix KV cache")
    p.add_argument("--tf-bf16", action="store_true",
                   help="run the transformer in bfloat16 (production dtype)")
    p.add_argument("-o", "--out",
                   default=os.path.join(tempfile.gettempdir(), "qwen21_ref"))
    return p.parse_args()


def main():
    a = parse_args()
    try:
        os.makedirs(a.out, exist_ok=True)
    except OSError as e:
        sys.exit(f"cannot create output dir {a.out}: {e}")

    print(f"Loading {a.dir} (float32, CPU) ...")
    try:
        pipe = QwenImage21Pipeline.from_pretrained(
            a.dir, torch_dtype=torch.float32, device_map="cpu")
    except Exception as e:
        sys.exit(f"failed to load {a.dir}: {e}")

    if a.tf_bf16:
        pipe.transformer = pipe.transformer.to(torch.bfloat16)
        print("transformer cast to bfloat16")

    state = {}

    # Capture the processor's input ids (needed for tokenizer parity).
    class _ProcCapture:
        def __init__(self, proc):
            self._proc = proc
        def __call__(self, **kw):
            out = self._proc(**kw)
            state["input_ids"] = out.input_ids[0].tolist()
            return out
        def __getattr__(self, name):
            return getattr(self._proc, name)
    pipe.processor = _ProcCapture(pipe.processor)

    # --- prompt encode: embeddings, masks ---------------------------------
    orig_encode = pipe.encode_prompt

    def hook_encode_prompt(*args, **kw):
        out = orig_encode(*args, **kw)  # orig_encode is bound to pipe
        state["embeds"] = out[0]
        state["enc_mask"] = out[1]
        state["img_mask"] = out[2]
        # Free the text encoder once its embeddings are captured: the denoise
        # loop needs only the transformer + VAE, and a float32 1024-sq run
        # otherwise blows past host RAM with encoder + transformer resident.
        if getattr(pipe, "text_encoder", None) is not None:
            pipe.text_encoder = None
            gc.collect()
        return out
    pipe.encode_prompt = hook_encode_prompt

    # --- latents: initial noise + reference latents ------------------------
    orig_prepare = pipe.prepare_latents

    def hook_prepare_latents(*args, **kw):
        out = orig_prepare(*args, **kw)  # bound to pipe
        state["latents0"] = out[0]
        state["ref_latents"] = out[1]
        return out
    pipe.prepare_latents = hook_prepare_latents

    # --- transformer: img_shapes, full img_mask, per-call velocity ---------
    import types
    orig_forward = pipe.transformer.forward

    def hook_transformer(self, *args, **kwargs):
        if a.tf_bf16:
            dt = next(self.parameters()).dtype
            args = tuple(t.to(dt) if torch.is_tensor(t) and t.is_floating_point() else t
                         for t in args)
            kwargs = {k: (t.to(dt) if torch.is_tensor(t) and t.is_floating_point() else t)
                      for k, t in kwargs.items()}
        out = orig_forward(*args, **kwargs)  # orig_forward bound to the module
        if not isinstance(out, tuple) or out[0] is None:
            return out
        if a.tf_bf16:
            out = (out[0].float(),) + tuple(out[1:])
        state["img_shapes"] = kwargs.get("img_shapes")
        state["img_mask_full"] = kwargs.get("img_mask")
        # Keep the same target-only slice the pipeline keeps: with
        # condition images the output covers [refs | target], and with the
        # KV cache it is target-only already.
        shapes = kwargs.get("img_shapes")
        if shapes:
            n_target = shapes[0][-1][1] * shapes[0][-1][2]
        else:
            n_target = out[0].shape[1]
        state["noise_preds"].append(out[0][:, -n_target:].clone())
        return out
    state["noise_preds"] = []
    pipe.transformer.forward = types.MethodType(hook_transformer, pipe.transformer)

    # --- vision tower: pixel_values, grid_thw, merged + deepstack embeds ----
    # Fires only for image-conditioned runs (pixel_values present). Gives the
    # C vision-tower implementation its per-stage parity targets.
    vis = getattr(getattr(pipe.text_encoder, "model", None), "visual", None)
    if vis is not None:
        orig_vis = vis.forward

        def hook_vis(self, hidden_states, grid_thw, *a, **kw):
            out = orig_vis(hidden_states, grid_thw, *a, **kw)
            state["vis_pixel_values"] = hidden_states
            state["vis_grid_thw"] = grid_thw
            state["vis_merged"] = out.pooler_output
            state["vis_deepstack"] = out.deepstack_features
            return out
        vis.forward = types.MethodType(hook_vis, vis)

    step_latents = []

    def on_step_end(pipe_, i, t, callback_kwargs):
        step_latents.append(callback_kwargs["latents"].clone())
        return callback_kwargs

    # --- run ---------------------------------------------------------------
    gen = torch.Generator("cpu").manual_seed(a.seed)
    extra = {}
    if a.noise:
        h_lat = (a.height // 32) * 2
        w_lat = (a.width // 32) * 2
        try:
            extra["latents"] = torch.from_numpy(
                np.fromfile(a.noise, dtype=np.float32)
                .reshape(1, h_lat * w_lat, 64)).clone()
        except (OSError, ValueError) as e:
            sys.exit(f"cannot load noise file {a.noise}: {e}")
    if a.ref:
        refs = []
        for p in a.ref:
            try:
                refs.append(Image.open(p))
            except OSError as e:
                sys.exit(f"cannot open reference image {p}: {e}")
        extra["image"] = refs

    print(f"Running: {a.width}x{a.height}, {a.steps} steps, "
          f"kv_cache={a.kv_cache}, refs={len(a.ref)} ...")
    out = pipe(
        prompt=a.prompt,
        width=a.width,
        height=a.height,
        num_inference_steps=a.steps,
        generator=gen,
        use_kv_cache=a.kv_cache,
        callback_on_step_end=on_step_end,
        output_type="pil",
        **extra,
    )
    out.images[0].save(os.path.join(a.out, "image.png"))
    # The scheduler keeps its final sigmas after the loop.
    state["sigmas"] = pipe.scheduler.sigmas.clone()

    # --- dump --------------------------------------------------------------
    embeds = state["embeds"][0]                      # [seq, 4096]
    drop = pipe._drop_idx
    h_lat, w_lat = (a.height // 16), (a.width // 16)

    write_text(a.out, "tokens.txt",
               "".join(f"{t}\n" for t in state.get("input_ids", [])))
    write_text(a.out, "drop_idx.txt", f"{drop}\n")
    dump_bin(a.out, "emb.bin", embeds)
    if state.get("enc_mask") is not None:
        dump_bin(a.out, "enc_mask.bin", state["enc_mask"][0])
    dump_bin(a.out, "img_mask.bin", state["img_mask"][0])
    dump_bin(a.out, "img_mask_full.bin", state["img_mask_full"][0])
    shapes = state["img_shapes"][0]
    write_text(a.out, "img_shapes.json",
               json.dumps([[s[0], s[1], s[2]] for s in shapes]))
    if state["ref_latents"] is not None:
        refs = state["ref_latents"][0]               # [total, 64]
        start = 0
        for i, s in enumerate(shapes[:-1]):
            n = s[1] * s[2]
            dump_bin(a.out, f"ref{i}.bin", refs[start:start + n])
            start += n
    if state.get("vis_pixel_values") is not None:
        dump_bin(a.out, "vis_pixel_values.bin", state["vis_pixel_values"])
        write_text(a.out, "vis_grid_thw.txt",
                   " ".join(str(int(x)) for x in state["vis_grid_thw"].flatten().tolist()) + "\n")
        dump_bin(a.out, "vis_merged.bin", state["vis_merged"])
        for i, df in enumerate(state["vis_deepstack"]):
            dump_bin(a.out, f"vis_deepstack{i}.bin", df)
    dump_bin(a.out, "noise.bin", state["latents0"][0])
    write_text(a.out, "sigmas.txt",
               "".join(f"{s:.9f}\n" for s in state["sigmas"].tolist()))
    for i, (v, x) in enumerate(zip(state["noise_preds"], step_latents,
                                   strict=True)):
        dump_bin(a.out, f"step{i}_v.bin", v[0])
        dump_bin(a.out, f"step{i}_x.bin", x[0])
    dump_bin(a.out, "latent.bin", step_latents[-1][0])

    print(f"Done. Wrote dumps + image.png to {a.out} ({h_lat}x{w_lat} latents)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
