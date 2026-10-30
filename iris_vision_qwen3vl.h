/*
 * Qwen3-VL vision tower for Iris (Qwen-Image 2.1 text encoder).
 *
 * Encodes reference images into merged embeddings consumed by the
 * Qwen-Image 2.1 DiT editing path. Weights are read from the text
 * encoder's safetensors shards under "model.visual." and kept in bf16
 * (mmap-backed); f32 is only per-operation scratch.
 */

#ifndef IRIS_VISION_QWEN3VL_H
#define IRIS_VISION_QWEN3VL_H

#include <stddef.h>

typedef struct qwv_model qwv_model_t;

/* Load the vision tower from a directory of safetensors shards.
 * grid_t/h/w describe one image: t frames, h*w/4 merged tokens. */
qwv_model_t *qwv_load(const char *dir);
void qwv_free(qwv_model_t *m);

/*
 * pixels: [t*h*w, 3*tp*16*16] float32, patches in merge-block order
 *         (block_row, block_col, in_row, in_col), values in [-1,1].
 * merged: caller-provided [h/2*w/2, out_hidden=4096].
 * deepstack_out: optional [3, h/2*w/2, 4096] (deepstack mergers of
 * blocks 8/16/24).
 * Returns 0 on success.
 */
int qwv_forward(qwv_model_t *m, int t, int h, int w, const float *pixels,
                float *merged, float *deepstack_out);

#endif /* IRIS_VISION_QWEN3VL_H */
