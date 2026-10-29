/*
 * Qwen-Image 2.1 Diffusion Transformer Implementation
 *
 * QwenImage21Transformer2DModel - single-stream DiT:
 * - 32 blocks, hidden 4096 (32 heads x 128 dim), SwiGLU FFN (12288)
 * - One shared AdaLN modulation (not per-block): temb -> 4*dim
 *   [mod1.scale, mod1.gate, mod2.scale, mod2.gate]
 * - Block-causal attention: (q_idx >= kv_idx) or same image block;
 *   right-padded text keys excluded
 * - causal_condition: text / condition-image tokens are modulated from
 *   t = 0 (two modulation rows: sampled timestep, then 0)
 * - 3-axis RoPE (frame, height, width) dims (16, 56, 56), theta 10000
 * - No bias parameters anywhere in the model
 *
 * Reference: diffusers/models/transformers/transformer_qwenimage21.py
 */

#include "iris.h"
#include "iris_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <glob.h>

/* Qwen-Image 2.1 Metal path is work-in-progress: the device forward faults in
 * the block loop (an MPSGraph fault at the second block's first GEMM). Until
 * that is fixed, this file uses the verified CPU path even on Metal builds
 * (IRIS_DEV). Define QW_ENABLE_METAL to compile and wire the device path. */
#if defined(IRIS_DEV) && !defined(QW_ENABLE_METAL)
#undef IRIS_DEV
#endif

#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

/* cblas_sgemm comes from the BLAS header for this build target: Accelerate on
 * Apple (IRIS_USE_ACCELERATE, set by the Makefile alongside
 * ACCELERATE_NEW_LAPACK), cblas.h under USE_OPENBLAS. */
#ifdef USE_BLAS
#if defined(IRIS_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
#include <cblas.h>
#endif
#endif

/* RoPE frequency table: positions 0..8191 plus negatives -1..-1024. */
#define QW_ROPE_ROWS 9216
#define QW_ROPE_MAX_NEG 1024

struct qw_transformer {
    int dim;         /* hidden dim (4096) */
    int heads;       /* attention heads (32) */
    int head_dim;    /* dim per head (128) */
    int n_layers;    /* blocks (32) */
    int mlp_dim;     /* FFN expansion (12288) */
    int in_ch;       /* latent channels (64) */
    int ctx_dim;     /* text encoder hidden dim (4096) */
    int axes[3];     /* RoPE axis dims (16, 56, 56) */
    float eps;
    int causal;      /* causal_condition */

    /* Big GEMM weights: held in their native bf16 on-disk format in memory
     * (~14 GB vs ~28 GB f32) to keep the loaded pipeline within RAM; widened
     * to f32 per-GEMM in qw_gemm (bf16->f32 is exact, so results are identical
     * to the all-f32 path). They are zero-copy pointers into the mmap'd
     * shards (get_bf16_direct) — see the shards field below. */
    uint16_t *img_in;    /* [dim, in_ch] */
    uint16_t *txt_in;    /* [dim, ctx_dim] */
    uint16_t *txt_out;   /* [dim, dim] */
    uint16_t *time_lin1; /* [dim, 256] */
    uint16_t *time_lin2; /* [dim, dim] */
    uint16_t *modulation; /* [4*dim, dim] */
    uint16_t *norm_out_lin; /* [dim, dim] */
    uint16_t *proj_out;  /* [in_ch, dim] */

    uint16_t **to_q, **to_k, **to_v, **to_out;  /* [n_layers] x [dim, dim] */
    uint16_t **mlp_gate, **mlp_proj, **mlp_out; /* [n_layers] x [mlp,dim]/[dim,mlp] */

    /* Small norm vectors: a few KB total, kept as f32 (qw_rmsnorm consumes them). */
    float *txt_norm_w; /* [ctx_dim], zero-centered */
    float **norm_q, **norm_k;                /* [n_layers] x [head_dim] */

    /* Per-axis RoPE tables: [9216][axis_dim] as (cos, sin) pairs. */
    float *rope[3];

    /* The safetensors shards stay open for the transformer's lifetime: the big
     * bf16 weights point into their mmap, so closing earlier would invalidate
     * them. Released in qw_transformer_free. */
    safetensors_file_t **shards;
    int nshards;

#ifdef IRIS_DEV
    struct qw_dev *dev;   /* device (Metal) copy of the model; NULL = CPU path */
#endif
};

typedef struct qw_transformer qw_transformer_t;

#ifdef IRIS_DEV
static int  qw_dev_load(qw_transformer_t *tf);
static int  qw_forward_dev(qw_transformer_t *tf, const float *z, const float *txt,
                           const int *txt_mask, int s_txt, int img_h, int img_w,
                           float sigma, float *v);
static void qw_dev_free(qw_transformer_t *tf);
static void qw_attn_host(qw_transformer_t *tf, const uint16_t *dqkv,
                         int s_txt, int n_img, const int *kv, float *dattn);
#endif

/* ========================================================================
 * Basic building blocks
 * ======================================================================== */

/* bf16 -> f32: a bf16 value is the upper 16 bits of an f32. */
static float qw_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, sizeof f); return f;
}

/* Bounded f32 scratch for widening one bf16 weight per GEMM. Grows to the
 * largest matrix (the MLP weights) and is reused across all GEMMs. */
static float *qw_wscratch = NULL;
static size_t qw_wscratch_cap = 0;

/* Row-major GEMM: out[n, o] = sum_k in[n, k] * W[o, k]. W stored [o, k].
 * W is bf16 (the on-disk format, kept in memory to halve the RAM footprint);
 * it is widened to f32 into a bounded scratch for the f32 GEMM. Because
 * bf16->f32 is exact, this reproduces the all-f32 path bit-for-bit. */
static void qw_gemm(const float *in, const uint16_t *W, float *out,
                    int n, int k, int o) {
    size_t nel = (size_t)k * o;
    if (nel > qw_wscratch_cap) {
        free(qw_wscratch);
        qw_wscratch = malloc(nel * sizeof(float));
        qw_wscratch_cap = nel;
    }
    float *Wf = qw_wscratch;
    for (size_t i = 0; i < nel; i++) Wf[i] = qw_bf16_to_f32(W[i]);
#ifdef USE_BLAS
    /* House BLAS convention (see iris_kernels.c iris_matmul_t): with B
     * transposed, ldb is the stored row stride of W, i.e. k. */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, o, k, 1.0f, in, k, Wf, k, 0.0f, out, o);
#else
    for (int j = 0; j < o; j++) {
        const float *w = Wf + (size_t)j * k;
        for (int i = 0; i < n; i++) {
            const float *x = in + (size_t)i * k;
            float acc = 0.0f;
            for (int p = 0; p < k; p++) acc += x[p] * w[p];
            out[(size_t)i * o + j] = acc;
        }
    }
#endif
}

/* LayerNorm without affine parameters, in-place (torch LayerNorm semantics:
 * biased variance + eps). */
static void qw_layernorm(float *x, int n, int d, float eps) {
    for (int i = 0; i < n; i++) {
        float *row = x + (size_t)i * d;
        float mean = 0.0f;
        for (int j = 0; j < d; j++) mean += row[j];
        mean /= d;
        float var = 0.0f;
        for (int j = 0; j < d; j++) {
            float t = row[j] - mean;
            var += t * t;
        }
        float rstd = 1.0f / sqrtf(var / d + eps);
        for (int j = 0; j < d; j++) row[j] = (row[j] - mean) * rstd;
    }
}

/* RMSNorm with learnable weight, in-place, computed in fp32.
 * zero_center: weight stored as (scale - 1). */
static void qw_rmsnorm(float *x, int n, int d, const float *w,
                       int zero_center, float eps) {
    for (int i = 0; i < n; i++) {
        float *row = x + (size_t)i * d;
        float ss = 0.0f;
        for (int j = 0; j < d; j++) ss += row[j] * row[j];
        float rstd = 1.0f / sqrtf(ss / d + eps);
        for (int j = 0; j < d; j++) {
            float scale = w[j];
            if (zero_center) scale += 1.0f;
            row[j] *= rstd * scale;
        }
    }
}

static void qw_gelu_tanh(float *x, int n) {
    const float c = 0.7978845608028654f; /* sqrt(2/pi) */
    const float g = 0.044715f;
    for (int i = 0; i < n; i++) {
        float t = x[i];
        x[i] = 0.5f * t * (1.0f + tanhf(c * (t + g * t * t * t)));
    }
}

static void qw_silu(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void qw_tanh(float *x, int n) {
    for (int i = 0; i < n; i++) x[i] = tanhf(x[i]);
}

/* ========================================================================
 * Weight loading
 * ======================================================================== */

/* f32 weight (small norm vectors); handles bf16/f16/f32 source. */
static float *qw_get_w(safetensors_file_t **shards, int nshards,
                       const char *name) {
    for (int i = 0; i < nshards; i++) {
        const safetensor_t *t = safetensors_find(shards[i], name);
        if (t) return safetensors_get_f32(shards[i], t);
    }
    return NULL;
}

/* bf16 weight (the big GEMM matrices): a zero-copy pointer into the mmap'd
 * shard, so the ~14 GB of weights stays file-backed (evictable) rather than a
 * separate f32/bf16 copy. Returns NULL if the tensor is not bf16. */
static uint16_t *qw_get_w_bf16(safetensors_file_t **shards, int nshards,
                               const char *name) {
    for (int i = 0; i < nshards; i++) {
        const safetensor_t *t = safetensors_find(shards[i], name);
        if (t && safetensor_is_bf16(t)) return safetensors_get_bf16_direct(shards[i], t);
    }
    return NULL;
}

qw_transformer_t *qw_transformer_load(const char *transformer_dir,
                                      int dim, int heads, int head_dim,
                                      int n_layers, int in_ch, int ctx_dim,
                                      int mlp_ratio,
                                      const int axes[3],
                                      float eps, int causal) {
    /* Open all safetensors shards in the transformer directory (the model
     * is split into diffusion_pytorch_model-0000N-of-0000M.safetensors). */
    char pattern[1024];
    snprintf(pattern, sizeof(pattern), "%s/*.safetensors", transformer_dir);
    glob_t g;
    if (glob(pattern, 0, NULL, &g) != 0) {
        fprintf(stderr, "Qwen-Image 2.1: no safetensors in %s\n", transformer_dir);
        return NULL;
    }
    if (g.gl_pathc < 1) {
        globfree(&g);
        return NULL;
    }
    safetensors_file_t **shards = malloc(g.gl_pathc * sizeof(*shards));
    int nshards = 0;
    for (long i = 0; i < g.gl_pathc; i++) {
        shards[nshards] = safetensors_open(g.gl_pathv[i]);
        if (!shards[nshards]) {
            fprintf(stderr, "Qwen-Image 2.1: cannot open %s\n", g.gl_pathv[i]);
            for (int j = 0; j < nshards; j++) safetensors_close(shards[j]);
            free(shards);
            globfree(&g);
            return NULL;
        }
        nshards++;
    }
    globfree(&g);

    qw_transformer_t *tf = calloc(1, sizeof(*tf));
    tf->dim = dim; tf->heads = heads; tf->head_dim = head_dim;
    tf->n_layers = n_layers; tf->mlp_dim = dim * mlp_ratio;
    tf->in_ch = in_ch; tf->ctx_dim = ctx_dim;
    tf->axes[0] = axes[0]; tf->axes[1] = axes[1]; tf->axes[2] = axes[2];
    tf->eps = eps; tf->causal = causal;

    tf->img_in = qw_get_w_bf16(shards, nshards, "img_in.weight");
    tf->txt_norm_w = qw_get_w(shards, nshards, "txt_in.text_norm.weight");
    tf->txt_in = qw_get_w_bf16(shards, nshards, "txt_in.in_layer.weight");
    tf->txt_out = qw_get_w_bf16(shards, nshards, "txt_in.out_layer.weight");
    tf->time_lin1 = qw_get_w_bf16(shards, nshards,
                             "time_text_embed.timestep_embedder.linear_1.weight");
    tf->time_lin2 = qw_get_w_bf16(shards, nshards,
                             "time_text_embed.timestep_embedder.linear_2.weight");
    /* modulation is a ModuleList: [0] = SiLU activation, [1] = Linear. */
    tf->modulation = qw_get_w_bf16(shards, nshards, "modulation.1.weight");
    tf->norm_out_lin = qw_get_w_bf16(shards, nshards, "norm_out.linear.weight");
    tf->proj_out = qw_get_w_bf16(shards, nshards, "proj_out.weight");
    if (!tf->img_in || !tf->txt_in || !tf->txt_out || !tf->time_lin1 ||
        !tf->time_lin2 || !tf->modulation || !tf->norm_out_lin ||
        !tf->proj_out || !tf->txt_norm_w) {
        fprintf(stderr, "Qwen-Image 2.1: missing/corrupt shared weight\n");
        free(tf);
        for (int i = 0; i < nshards; i++) safetensors_close(shards[i]);
        free(shards);
        return NULL;
    }

    const char *pref = "transformer_blocks.";
    char name[256];
    tf->to_q = malloc(n_layers * sizeof(*tf->to_q));
    tf->to_k = malloc(n_layers * sizeof(*tf->to_k));
    tf->to_v = malloc(n_layers * sizeof(*tf->to_v));
    tf->to_out = malloc(n_layers * sizeof(*tf->to_out));
    tf->mlp_gate = malloc(n_layers * sizeof(*tf->mlp_gate));
    tf->mlp_proj = malloc(n_layers * sizeof(*tf->mlp_proj));
    tf->mlp_out = malloc(n_layers * sizeof(*tf->mlp_out));
    tf->norm_q = malloc(n_layers * sizeof(*tf->norm_q));
    tf->norm_k = malloc(n_layers * sizeof(*tf->norm_k));
    /* 7 big GEMM weights (bf16) + 2 small norm vectors (f32) per block. */
    static const char *sfx[] = {
        ".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight",
        ".attn.to_out.0.weight", ".img_mlp.gate_layer.weight",
        ".img_mlp.proj.weight", ".img_mlp.out.weight",
        ".attn.norm_q.weight", ".attn.norm_k.weight"
    };
    uint16_t **gemms[7] = { tf->to_q, tf->to_k, tf->to_v, tf->to_out,
                            tf->mlp_gate, tf->mlp_proj, tf->mlp_out };
    float **norms[2] = { tf->norm_q, tf->norm_k };
    for (int l = 0; l < n_layers; l++) {
        for (int w = 0; w < 9; w++) {
            snprintf(name, sizeof(name), "%s%d%s", pref, l, sfx[w]);
            void *got = (w < 7)
                ? (void *)qw_get_w_bf16(shards, nshards, name)
                : (void *)qw_get_w(shards, nshards, name);
            if (w < 7) gemms[w][l] = (uint16_t *)got;
            else       norms[w - 7][l] = (float *)got;
            if (!got) {
                fprintf(stderr, "Qwen-Image 2.1: missing weight %s\n", name);
                free(tf);
                for (int i = 0; i < nshards; i++) safetensors_close(shards[i]);
                free(shards);
                return NULL;
            }
        }
    }

    /* Keep the shards open: the big bf16 weights point into their mmap
     * (get_bf16_direct); released in qw_transformer_free. */
    tf->shards = shards;
    tf->nshards = nshards;

    /* RoPE frequency tables. For each axis of dim d: pair j (j < d/2) has
     * angle idx * theta^(-2j/d). Rows 0..8191 encode positions 0..8191;
     * rows 8192..9215 encode -1..-1024. */
    for (int a = 0; a < 3; a++) {
        int d = tf->axes[a];
        int pairs = d / 2;
        tf->rope[a] = malloc((size_t)QW_ROPE_ROWS * d * sizeof(float));
        for (int r = 0; r < QW_ROPE_ROWS; r++) {
            /* Rows 0..8191: positions 0..8191. Rows 8192..9215: -1024..-1
             * (matches diffusers' neg_index = [-1024, ..., -1]). */
            int idx = (r < 8192) ? r : (r - QW_ROPE_ROWS);
            for (int j = 0; j < pairs; j++) {
                float angle = (float)idx * powf(10000.0f, -2.0f * j / d);
                tf->rope[a][(size_t)r * d + 2 * j] = cosf(angle);
                tf->rope[a][(size_t)r * d + 2 * j + 1] = sinf(angle);
            }
        }
    }

    printf("Qwen-Image 2.1 transformer loaded: %d layers, dim %d, "
           "%d heads x %d, mlp %d\n",
           n_layers, dim, heads, head_dim, dim * mlp_ratio);
#ifdef IRIS_DEV
    if (iris_dev_available()) qw_dev_load(tf);
#endif
    return tf;
}

void qw_transformer_free(qw_transformer_t *tf) {
    if (!tf) return;
#ifdef IRIS_DEV
    qw_dev_free(tf);
#endif
    /* The big bf16 GEMM weights are zero-copy pointers into the shard mmap:
     * closing the files releases them (do not free the pointers). Only the
     * small norm vectors are malloc'd f32 and are freed here. */
    free(tf->txt_norm_w);
    for (int l = 0; l < tf->n_layers; l++) {
        free(tf->norm_q[l]); free(tf->norm_k[l]);
    }
    for (int i = 0; i < tf->nshards; i++) safetensors_close(tf->shards[i]);
    free(tf->shards);
    free(tf->to_q); free(tf->to_k); free(tf->to_v); free(tf->to_out);
    free(tf->mlp_gate); free(tf->mlp_proj); free(tf->mlp_out);
    free(tf->norm_q); free(tf->norm_k);
    for (int a = 0; a < 3; a++) free(tf->rope[a]);
    free(tf);
}

/* ========================================================================
 * Timestep embedding and shared modulation
 * ======================================================================== */

/* Sinusoidal embedding: [cos(t*f_j), sin(t*f_j)], f_j = 10000^(-j/128). */
static void qw_time_embed(float *emb, float t) {
    for (int j = 0; j < 128; j++) {
        float f = expf(-logf(10000.0f) * j / 128);
        emb[j] = cosf(t * f);
        emb[128 + j] = sinf(t * f);
    }
}

/* RoPE lookup row for a (possibly negative) position index. */
static int qw_rope_row(int idx) {
    return idx >= 0 ? idx : QW_ROPE_ROWS + idx;
}

/* Fill a token's RoPE table indexed by head dim: c[2j]/s[2j] hold the
 * cos/sin of global pair j (apply reads tc[p] at even head dim p = 2j).
 * Pairs 0..7 are the frame axis, 8..35 height, 36..63 width — the
 * reference's cat of the three axis freq tables. */
static void qw_token_rope(qw_transformer_t *tf, int frame, int h, int w,
                          float *c, float *s) {
    float *t;
    for (int j = 0; j < 8; j++) {
        t = tf->rope[0] + (size_t)qw_rope_row(frame) * tf->axes[0] + 2 * j;
        c[2 * j] = t[0]; s[2 * j] = t[1];
    }
    for (int j = 8; j < 36; j++) {
        t = tf->rope[1] + (size_t)qw_rope_row(h) * tf->axes[1] + 2 * (j - 8);
        c[2 * j] = t[0]; s[2 * j] = t[1];
    }
    for (int j = 36; j < 64; j++) {
        t = tf->rope[2] + (size_t)qw_rope_row(w) * tf->axes[2] + 2 * (j - 36);
        c[2 * j] = t[0]; s[2 * j] = t[1];
    }
}

/* Apply RoPE to [n_tok, heads, head_dim] q or k (complex-pair convention:
 * x'[2j] = x[2j]*cos - x[2j+1]*sin; x'[2j+1] = x[2j]*sin + x[2j+1]*cos). */
static void qw_rope_apply(float *x, int n_tok, int heads, int hd,
                          const float *c, const float *s) {
    for (int i = 0; i < n_tok; i++) {
        const float *tc = c + (size_t)i * hd;
        const float *ts = s + (size_t)i * hd;
        for (int h = 0; h < heads; h++) {
            float *row = x + ((size_t)i * heads + h) * hd;
#ifdef QW_ROPE_DEBUG
            if (i == 1 && h == 0 && n_tok > 1) {
                fprintf(stderr, "  rope_apply: n_tok=%d tc[0]=%.6g ts[0]=%.6g row[0] %.6g -> ",
                        n_tok, tc[0], ts[0], row[0]);
            }
#endif
            for (int p = 0; p < hd; p += 2) {
                float a = row[p], b = row[p + 1];
                row[p] = a * tc[p] - b * ts[p];
                row[p + 1] = a * ts[p] + b * tc[p];
            }
#ifdef QW_ROPE_DEBUG
            if (i == 1 && h == 0 && n_tok > 1)
                fprintf(stderr, "%.6g\n", row[0]);
#endif
        }
    }
}

/* One attention segment of the block-causal prefill.
 * q/k/v: [n_all, heads, head_dim]; queries are rows [q_start, q_end),
 * keys rows [0, kv_end). Text segments add a causal triangle
 * (q_abs >= kv_abs); image segments see their whole key range. In either
 * case padded (key_valid == 0) keys are excluded.
 * USE_BLAS builds use batched GEMMs for the two matmuls; the mask and
 * softmax are identical in both paths. */
static void qw_seg_attn(const float *q, const float *k, const float *v,
                        int heads, int hd, int q_start, int q_end,
                        int kv_end, int is_text, const int *key_valid,
                        float *out) {
    int nq = q_end - q_start;
    float *scores = malloc((size_t)nq * kv_end * sizeof(float));
    float scale = 1.0f / sqrtf((float)hd);
    for (int h = 0; h < heads; h++) {
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    nq, kv_end, hd, scale,
                    q + ((size_t)q_start * heads + h) * hd, heads * hd,
                    k + h * hd, heads * hd, 0.0f, scores, kv_end);
        for (int i = 0; i < nq; i++) {
            float *sr = scores + (size_t)i * kv_end;
            int q_abs = q_start + i;
            for (int j = 0; j < kv_end; j++) {
                if (!key_valid[j] || (is_text && q_abs < j))
                    sr[j] = -INFINITY;
            }
            float mx = -INFINITY;
            for (int j = 0; j < kv_end; j++) if (sr[j] > mx) mx = sr[j];
            if (mx == -INFINITY) { /* all-invalid rows stay zero */
                memset(sr, 0, (size_t)kv_end * sizeof(float));
                continue;
            }
            float sum = 0.0f;
            for (int j = 0; j < kv_end; j++) {
                sr[j] = expf(sr[j] - mx);
                sum += sr[j];
            }
            for (int j = 0; j < kv_end; j++) sr[j] /= sum;
        }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    nq, hd, kv_end, 1.0f,
                    scores, kv_end,
                    v + h * hd, heads * hd, 0.0f,
                    out + ((size_t)q_start * heads + h) * hd, heads * hd);
#else
        for (int i = 0; i < nq; i++) {
            const float *qr = q + ((size_t)(q_start + i) * heads + h) * hd;
            float *sr = scores + (size_t)i * kv_end;
            for (int j = 0; j < kv_end; j++) {
                const float *kr = k + ((size_t)j * heads + h) * hd;
                float s = 0.0f;
                for (int p = 0; p < hd; p++) s += qr[p] * kr[p];
                sr[j] = s * scale;
            }
        }
        for (int i = 0; i < nq; i++) {
            float *sr = scores + (size_t)i * kv_end;
            int q_abs = q_start + i;
            float mx = -INFINITY;
            for (int j = 0; j < kv_end; j++) {
                int ok = key_valid[j] && (is_text ? q_abs >= j : 1);
                if (!ok) { sr[j] = -INFINITY; continue; }
                if (sr[j] > mx) mx = sr[j];
            }
            if (mx == -INFINITY) { /* never happens in practice: valid keys
                                     precede every query; keep rows finite */
                memset(sr, 0, (size_t)kv_end * sizeof(float));
                continue;
            }
            float sum = 0.0f;
            for (int j = 0; j < kv_end; j++) {
                sr[j] = expf(sr[j] - mx);
                sum += sr[j];
            }
            for (int j = 0; j < kv_end; j++) sr[j] /= sum;
        }
        for (int i = 0; i < nq; i++) {
            float *orow = out + ((size_t)(q_start + i) * heads + h) * hd;
            const float *sr = scores + (size_t)i * kv_end;
            for (int p = 0; p < hd; p++) orow[p] = 0.0f;
            for (int j = 0; j < kv_end; j++) {
                float s = sr[j];
                if (s == 0.0f) continue;
                const float *vrow = v + ((size_t)j * heads + h) * hd;
                for (int p = 0; p < hd; p++) orow[p] += s * vrow[p];
            }
        }
#endif
    }
    free(scores);
}

/* One single-stream block:
 * x += tanh(g1) * attn(norm1(x) * (1 + s1));
 * x += tanh(g2) * mlp(norm2(x) * (1 + s2));
 * Modulation rows are selected per token: target tokens use the sampled-
 * timestep row, text (and condition-image) tokens use the t=0 row.
 * x: in/out [N, dim]; xm: scratch [N, dim];
 * mT/m0: the two modulation rows [4*dim];
 * is_target: [N] 0/1 (1 = target-image tokens). */
static void qw_block(qw_transformer_t *tf, int b, float *x, float *xm,
                     int n_tok, const float *mT, const float *m0,
                     const int *is_target,
                     const float *qk_c, const float *qk_s,  /* per-token RoPE */
                     int s_txt, int N, const int *key_valid)
{
    int dim = tf->dim, heads = tf->heads, hd = tf->head_dim;
    int d4 = 4 * dim;
    const float *sc1T = mT, *g1T = mT + dim, *sc2T = mT + 2 * dim, *g2T = mT + 3 * dim;
    const float *sc10 = m0, *g10 = m0 + dim, *sc20 = m0 + 2 * dim, *g20 = m0 + 3 * dim;

    /* norm1 + modulate 1 */
    memcpy(xm, x, (size_t)n_tok * dim * sizeof(float));
    qw_layernorm(xm, n_tok, dim, tf->eps);
    for (int i = 0; i < n_tok; i++) {
        const float *sc = is_target[i] ? sc1T : sc10;
        float *row = xm + (size_t)i * dim;
        for (int j = 0; j < dim; j++) row[j] *= 1.0f + sc[j];
    }
    /* attention */
    float *q = malloc((size_t)n_tok * dim * sizeof(float));
    float *k = malloc((size_t)n_tok * dim * sizeof(float));
    float *vv = malloc((size_t)n_tok * dim * sizeof(float));
    qw_gemm(xm, tf->to_q[b], q, n_tok, dim, dim);
    qw_gemm(xm, tf->to_k[b], k, n_tok, dim, dim);
    qw_gemm(xm, tf->to_v[b], vv, n_tok, dim, dim);
    for (int i = 0; i < n_tok; i++) {
        for (int h = 0; h < heads; h++) {
            qw_rmsnorm(q + ((size_t)i * heads + h) * hd, 1, hd,
                       tf->norm_q[b], 0, tf->eps);
            qw_rmsnorm(k + ((size_t)i * heads + h) * hd, 1, hd,
                       tf->norm_k[b], 0, tf->eps);
        }
    }
    qw_rope_apply(q, n_tok, heads, hd, qk_c, qk_s);
    qw_rope_apply(k, n_tok, heads, hd, qk_c, qk_s);

    float *attnout = malloc((size_t)n_tok * dim * sizeof(float));
    if (s_txt > 0) {
        qw_seg_attn(q, k, vv, heads, hd, 0, s_txt, s_txt, 1, key_valid, attnout);
    }
    qw_seg_attn(q, k, vv, heads, hd, s_txt, N, N, 0, key_valid, attnout);
    free(q); free(k); free(vv);

    float *a = malloc((size_t)n_tok * dim * sizeof(float));
    qw_gemm(attnout, tf->to_out[b], a, n_tok, dim, dim);
    free(attnout);

    /* gated residual 1 */
    for (int i = 0; i < n_tok; i++) {
        const float *g = is_target[i] ? g1T : g10;
        const float *ar = a + (size_t)i * dim;
        float *xr = x + (size_t)i * dim;
        for (int j = 0; j < dim; j++) xr[j] += tanhf(g[j]) * ar[j];
    }
    free(a);

    /* norm2 + modulate 2, SwiGLU FFN */
    memcpy(xm, x, (size_t)n_tok * dim * sizeof(float));
    qw_layernorm(xm, n_tok, dim, tf->eps);
    for (int i = 0; i < n_tok; i++) {
        const float *sc = is_target[i] ? sc2T : sc20;
        float *row = xm + (size_t)i * dim;
        for (int j = 0; j < dim; j++) row[j] *= 1.0f + sc[j];
    }
    float *gate = malloc((size_t)n_tok * tf->mlp_dim * sizeof(float));
    float *up = malloc((size_t)n_tok * tf->mlp_dim * sizeof(float));
    qw_gemm(xm, tf->mlp_gate[b], gate, n_tok, dim, tf->mlp_dim);
    qw_gemm(xm, tf->mlp_proj[b], up, n_tok, dim, tf->mlp_dim);
    qw_silu(gate, n_tok * tf->mlp_dim);
    for (int i = 0; i < n_tok * tf->mlp_dim; i++) gate[i] *= up[i];
    free(up);
    float *h = malloc((size_t)n_tok * dim * sizeof(float));
    qw_gemm(gate, tf->mlp_out[b], h, n_tok, tf->mlp_dim, dim);
    free(gate);
    for (int i = 0; i < n_tok; i++) {
        const float *g = is_target[i] ? g2T : g20;
        const float *hr = h + (size_t)i * dim;
        float *xr = x + (size_t)i * dim;
        for (int j = 0; j < dim; j++) xr[j] += tanhf(g[j]) * hr[j];
    }
    free(h);
    (void)d4;
}

/* ========================================================================
 * Forward pass (t2i: one target block, no KV cache)
 * ======================================================================== */

/*
 * Compute the velocity prediction for the target image.
 *
 * z:          [n_img * in_ch] target latents (n_img = img_h * img_w)
 * txt:        [s_txt * ctx_dim] VLM text embeddings (right-padded)
 * txt_mask:   [s_txt] 0/1 validity per text token (NULL = all valid)
 * img_h/w:    target block in latent tokens
 * sigma:      current flow-matching sigma (0..1)
 * v:          [n_img * in_ch] predicted velocity (output)
 */
#ifdef IRIS_DEV
/* ========================================================================
 * GPU forward (Metal). Weights are uploaded once as bf16 (their native
 * on-disk format); activations run bf16 with f32 accumulation. The block
 * math mirrors qw_block(); block-causal attention runs as two calls: the
 * text segment causal over text keys, the image segment non-causal over all
 * keys, both gated by the per-token key-validity mask.
 * ======================================================================== */

typedef struct {
    uint16_t *to_q, *to_k, *to_v, *to_out;   /* [dim, dim] */
    uint16_t *mlp_gate, *mlp_proj, *mlp_out; /* gate/up [mlp,dim], out [dim,mlp] */
    float *norm_q, *norm_k;                  /* [head_dim] */
} qw_dev_block_t;

struct qw_dev {
    uint16_t *img_in, *txt_in, *txt_out, *time_lin1, *time_lin2;
    uint16_t *modulation, *norm_out_lin, *proj_out;
    float *txt_norm_w;                        /* [ctx_dim] zero-centered */
    qw_dev_block_t *blocks;
    /* modulation-row layout offsets within vec / vec_b16 (in elements). */
    int o_sin, o_h1, o_act, o_temb, o_mod, o_sout, o_tg, vecN;
    /* activations, sized for seq tokens */
    int seq;
    float *x;             /* [seq, dim] residual stream (f32) */
    uint16_t *norm;       /* [seq, dim] modulated LayerNorm (bf16) */
    uint16_t *qkv;        /* [seq, 3*dim] */
    uint16_t *gate;       /* [seq, mlp_dim] */
    uint16_t *ffn;        /* [seq, mlp_dim] */
    uint16_t *att;        /* [seq, dim] attention output */
    uint16_t *cat;        /* [seq, dim] (unused, kept simple) */
    float *o1, *o2;       /* [seq, dim] attention/FFN projection scratch (f32) */
    float *rope_c, *rope_s;  /* [seq, head_dim/2] */
    int *mask;            /* [seq] key validity */
    float *vec;           /* modulation temb/mod/scale rows (f32) */
    uint16_t *vec_b16;
    /* model input staging */
    size_t in_elems;
    float *in_f32;
    uint16_t *in_b16;
};

static void qw_dev_free(qw_transformer_t *tf) {
    struct qw_dev *d = tf->dev;
    if (!d) return;
    if (d->blocks) {
        for (int l = 0; l < tf->n_layers; l++) {
            qw_dev_block_t *b = &d->blocks[l];
            iris_dev_free(b->to_q); iris_dev_free(b->to_k);
            iris_dev_free(b->to_v); iris_dev_free(b->to_out);
            iris_dev_free(b->mlp_gate); iris_dev_free(b->mlp_proj); iris_dev_free(b->mlp_out);
            iris_dev_free(b->norm_q); iris_dev_free(b->norm_k);
        }
        free(d->blocks);
    }
    iris_dev_free(d->img_in); iris_dev_free(d->txt_in); iris_dev_free(d->txt_out);
    iris_dev_free(d->time_lin1); iris_dev_free(d->time_lin2);
    iris_dev_free(d->modulation); iris_dev_free(d->norm_out_lin); iris_dev_free(d->proj_out);
    iris_dev_free(d->txt_norm_w);
    iris_dev_free(d->x); iris_dev_free(d->norm); iris_dev_free(d->qkv);
    iris_dev_free(d->gate); iris_dev_free(d->ffn); iris_dev_free(d->att); iris_dev_free(d->cat);
    iris_dev_free(d->o1); iris_dev_free(d->o2);
    iris_dev_free(d->rope_c); iris_dev_free(d->rope_s); iris_dev_free(d->mask);
    iris_dev_free(d->vec); iris_dev_free(d->vec_b16);
    iris_dev_free(d->in_f32); iris_dev_free(d->in_b16);
    free(d);
    tf->dev = NULL;
}

static uint16_t *dev_up_bf16(const uint16_t *src, size_t n) {
    uint16_t *d = iris_dev_alloc(n * sizeof(uint16_t));
    if (d) iris_dev_upload(d, src, n * sizeof(uint16_t));
    return d;
}
static float *dev_up_f32(const float *src, size_t n) {
    float *d = iris_dev_alloc(n * sizeof(float));
    if (d) iris_dev_upload(d, src, n * sizeof(float));
    return d;
}

static int qw_dev_load(qw_transformer_t *tf) {
    if (!iris_dev_available()) return 0;
    struct qw_dev *d = calloc(1, sizeof(*d));
    if (!d) return 0;
    tf->dev = d;
    int dim = tf->dim, mlp = tf->mlp_dim, hd = tf->head_dim, nl = tf->n_layers;
    size_t ok = 1;

    d->img_in = dev_up_bf16(tf->img_in, (size_t)dim * tf->in_ch);
    d->txt_in = dev_up_bf16(tf->txt_in, (size_t)dim * tf->ctx_dim);
    d->txt_out = dev_up_bf16(tf->txt_out, (size_t)dim * dim);
    d->time_lin1 = dev_up_bf16(tf->time_lin1, (size_t)dim * 256);
    d->time_lin2 = dev_up_bf16(tf->time_lin2, (size_t)dim * dim);
    d->modulation = dev_up_bf16(tf->modulation, (size_t)4 * dim * dim);
    d->norm_out_lin = dev_up_bf16(tf->norm_out_lin, (size_t)dim * dim);
    d->proj_out = dev_up_bf16(tf->proj_out, (size_t)tf->in_ch * dim);
    d->txt_norm_w = dev_up_f32(tf->txt_norm_w, tf->ctx_dim);
    ok &= d->img_in && d->txt_in && d->txt_out && d->time_lin1 && d->time_lin2 &&
          d->modulation && d->norm_out_lin && d->proj_out && d->txt_norm_w;

    d->blocks = ok ? calloc(nl, sizeof(qw_dev_block_t)) : NULL;
    ok &= d->blocks != NULL;
    for (int l = 0; ok && l < nl; l++) {
        qw_dev_block_t *b = &d->blocks[l];
        b->to_q = dev_up_bf16(tf->to_q[l], (size_t)dim * dim);
        b->to_k = dev_up_bf16(tf->to_k[l], (size_t)dim * dim);
        b->to_v = dev_up_bf16(tf->to_v[l], (size_t)dim * dim);
        b->to_out = dev_up_bf16(tf->to_out[l], (size_t)dim * dim);
        b->mlp_gate = dev_up_bf16(tf->mlp_gate[l], (size_t)mlp * dim);
        b->mlp_proj = dev_up_bf16(tf->mlp_proj[l], (size_t)mlp * dim);
        b->mlp_out = dev_up_bf16(tf->mlp_out[l], (size_t)dim * mlp);
        b->norm_q = dev_up_f32(tf->norm_q[l], hd);
        b->norm_k = dev_up_f32(tf->norm_k[l], hd);
        ok &= b->to_q && b->to_k && b->to_v && b->to_out && b->mlp_gate &&
              b->mlp_proj && b->mlp_out && b->norm_q && b->norm_k;
    }

    /* Modulation-row scratch: one vector holding tembT/temb0, sinusoid inputs,
     * hidden, silu(temb) (bf16), modulation, output scale, and tanh'd gates. */
    d->o_sin = 0;              /* sinT, sin0            [256 each] */
    d->o_h1  = d->o_sin + 256; /* h1T, h10             [dim each]  */
    d->o_act = d->o_h1 + 2 * dim;   /* actT, act0       [dim each]  */
    d->o_temb = d->o_act + 2 * dim; /* tembT, temb0     [dim each]  */
    d->o_mod  = d->o_temb + 2 * dim;/* modT, mod0       [4dim each] */
    d->o_sout = d->o_mod + 8 * dim; /* soutT, sout0     [dim each]  */
    d->o_tg   = d->o_sout + 2 * dim;/* g1T,g2T,g10,g20  [dim each]  */
    d->vecN   = d->o_tg + 4 * dim;
    d->vec = ok ? iris_dev_alloc((size_t)d->vecN * sizeof(float)) : NULL;
    d->vec_b16 = ok ? iris_dev_alloc((size_t)d->vecN * sizeof(uint16_t)) : NULL;
    ok &= d->vec && d->vec_b16;

    if (!ok) {
        fprintf(stderr, "Qwen-Image 2.1: not enough GPU memory, using the CPU\n");
        qw_dev_free(tf);
        return 0;
    }
    iris_dev_sync();
    return 1;
}

static int qw_dev_ensure(qw_transformer_t *tf, int seq, size_t in_elems) {
    struct qw_dev *d = tf->dev;
    int dim = tf->dim, mlp = tf->mlp_dim, hd = tf->head_dim;
    if (seq > d->seq) {
        iris_dev_free(d->x); iris_dev_free(d->norm); iris_dev_free(d->qkv);
        iris_dev_free(d->gate); iris_dev_free(d->ffn); iris_dev_free(d->att);
        iris_dev_free(d->o1); iris_dev_free(d->o2);
        iris_dev_free(d->rope_c); iris_dev_free(d->rope_s); iris_dev_free(d->mask);
        d->x   = iris_dev_alloc((size_t)seq * dim * sizeof(float));
        d->norm= iris_dev_alloc((size_t)seq * dim * sizeof(uint16_t));
        d->qkv = iris_dev_alloc((size_t)seq * 3 * dim * sizeof(uint16_t));
        d->gate= iris_dev_alloc((size_t)seq * mlp * sizeof(uint16_t));
        d->ffn = iris_dev_alloc((size_t)seq * mlp * sizeof(uint16_t));
        d->att = iris_dev_alloc((size_t)seq * dim * sizeof(uint16_t));
        d->o1  = iris_dev_alloc((size_t)seq * dim * sizeof(float));
        d->o2  = iris_dev_alloc((size_t)seq * dim * sizeof(float));
        d->rope_c = iris_dev_alloc((size_t)seq * hd / 2 * sizeof(float));
        d->rope_s = iris_dev_alloc((size_t)seq * hd / 2 * sizeof(float));
        d->mask = iris_dev_alloc((size_t)seq * sizeof(int));
        if (!d->x || !d->norm || !d->qkv || !d->gate || !d->ffn || !d->att ||
            !d->o1 || !d->o2 || !d->rope_c || !d->rope_s || !d->mask) return 0;
        d->seq = seq;
    }
    if (in_elems > d->in_elems) {
        iris_dev_free(d->in_f32); iris_dev_free(d->in_b16);
        d->in_f32 = iris_dev_alloc(in_elems * sizeof(float));
        d->in_b16 = iris_dev_alloc(in_elems * sizeof(uint16_t));
        d->in_elems = (d->in_f32 && d->in_b16) ? in_elems : 0;
        if (!d->in_elems) return 0;
    }
    return 1;
}

/* Metal's iris_dev_attention is non-causal only. The text segment (<= 512
 * tokens, and small relative to the image) runs causal attention on the host
 * by downloading its q/k/v rows, so the whole image block stays on-device. */
/* Block-causal attention on the host. The fused SDPA graph faults in this
 * build, so q/k/v (already computed on-device, bf16) are downloaded, the two
 * segments run through the CPU-verified qw_seg_attn, and the result is
 * uploaded. Only the small attention tensor crosses; GEMMs stay on-device. */
/* Text-segment attention on the host. Device SDPA is non-causal only, so the
 * small causal text block (few dozen tokens) runs through the CPU-verified
 * qw_seg_attn and the result is uploaded as bf16 (the block reads d->att via
 * linear_f32, which expects a bf16 operand). The large image segment runs
 * on-device via iris_dev_attention. */
static void qw_attn_host(qw_transformer_t *tf, const uint16_t *dqkv,
                         int s_txt, int n_img, const int *kv, float *dattn) {
    (void)n_img; (void)dattn;
    if (s_txt <= 0) return;
    int heads = tf->heads, hd = tf->head_dim, dim = tf->dim;
    size_t sz = (size_t)s_txt * dim;      /* text rows only */
    iris_dev_sync();
    float *q = malloc(sz * sizeof(float));
    float *k = malloc(sz * sizeof(float));
    float *v = malloc(sz * sizeof(float));
    float *hout = malloc(sz * sizeof(float));
    uint16_t *bout_unused = NULL; (void)bout_unused;
    /* Device buffers are CPU-mapped; read qkv in place (the large-copy
     * download path returns zeros for big buffers here). */
    for (size_t i = 0; i < sz; i++) {
        q[i] = qw_bf16_to_f32(dqkv[i]);
        k[i] = qw_bf16_to_f32(dqkv[dim + i]);
        v[i] = qw_bf16_to_f32(dqkv[2 * dim + i]);
    }
    qw_seg_attn(q, k, v, heads, hd, 0, s_txt, s_txt, 1, kv, hout);
    /* Reuse the text q/k/v slots as bf16 scratch (>= s_txt*dim). */
    uint16_t *b16 = (uint16_t *)q;
    size_t m = 0;
    for (size_t i = 0; i < sz; i++) {
        uint32_t u; memcpy(&u, &hout[i], 4);
        u += 0x8000u;                       /* round-to-nearest */
        b16[m++] = (uint16_t)(u >> 16);
    }
    iris_dev_upload(dattn, b16, sz * sizeof(uint16_t));
    free(q); free(k); free(v); free(hout);
}

static int qw_forward_dev(qw_transformer_t *tf, const float *z, const float *txt,
                          const int *txt_mask, int s_txt, int img_h, int img_w,
                          float sigma, float *v) {
    struct qw_dev *d = tf->dev;
    int dim = tf->dim, heads = tf->heads, hd = tf->head_dim, half = hd / 2;
    int mlp = tf->mlp_dim, n_img = img_h * img_w, N = s_txt + n_img;
    float t = 1000.0f * sigma;

    size_t in_elems = (size_t)s_txt * tf->ctx_dim;
    if ((size_t)n_img * tf->in_ch > in_elems) in_elems = (size_t)n_img * tf->in_ch;
    if ((size_t)(s_txt > 0 ? s_txt : n_img) * dim > in_elems)
        in_elems = (size_t)(s_txt > 0 ? s_txt : n_img) * dim;
    if (!qw_dev_ensure(tf, N, in_elems)) {
        fprintf(stderr, "Qwen-Image 2.1: GPU allocation failed for %d tokens\n", N);
        return 0;
    }

    /* Text projection on the host (only a few dozen tokens; needs the
     * zero-centered RMSNorm and GELU that the device API lacks). Result rows
     * are uploaded to x. */
    if (s_txt > 0) {
        float *enc = malloc((size_t)s_txt * tf->ctx_dim * sizeof(float));
        memcpy(enc, txt, (size_t)s_txt * tf->ctx_dim * sizeof(float));
        qw_rmsnorm(enc, s_txt, tf->ctx_dim, tf->txt_norm_w, 1, tf->eps);
        /* Every graph input/output must be an iris_dev_alloc'd buffer (the
         * Metal backend maps pointers back to MTLBuffers); host round-trips
         * (for GELU) go through download/upload, never a malloc'd buffer. */
        iris_dev_upload(d->in_f32, enc, (size_t)s_txt * tf->ctx_dim * sizeof(float));
        iris_dev_f32_to_bf16(d->in_b16, d->in_f32, (size_t)s_txt * tf->ctx_dim);
        iris_dev_linear_f32(d->o1, d->in_b16, d->txt_in, s_txt, tf->ctx_dim, dim);
        float *pin = malloc((size_t)s_txt * dim * sizeof(float));
        iris_dev_download(pin, d->o1, (size_t)s_txt * dim * sizeof(float));
        qw_gelu_tanh(pin, s_txt * dim);
        iris_dev_upload(d->in_f32, pin, (size_t)s_txt * dim * sizeof(float));
        iris_dev_f32_to_bf16(d->in_b16, d->in_f32, (size_t)s_txt * dim);
        free(pin); free(enc);
        iris_dev_linear_f32(d->x, d->in_b16, d->txt_out, s_txt, dim, dim);
    }

    /* Image latents -> x[s_txt:]. */
    iris_dev_upload(d->in_f32, z, (size_t)n_img * tf->in_ch * sizeof(float));
    iris_dev_f32_to_bf16(d->in_b16, d->in_f32, (size_t)n_img * tf->in_ch);
    iris_dev_linear_f32(d->x + (size_t)s_txt * dim, d->in_b16, d->img_in,
                        n_img, tf->in_ch, dim);

    /* Timestep embeddings for sigma and t = 0. */
    float embT[256], emb0[256];
    qw_time_embed(embT, t);
    for (int j = 0; j < 128; j++) { emb0[j] = 1.0f; emb0[128 + j] = 0.0f; }
    iris_dev_upload(d->vec + d->o_sin, embT, 256 * sizeof(float));
    iris_dev_upload(d->vec + d->o_sin + 256, emb0, 256 * sizeof(float));
    iris_dev_f32_to_bf16(d->vec_b16 + d->o_sin, d->vec + d->o_sin, 256);
    iris_dev_f32_to_bf16(d->vec_b16 + d->o_sin + 256, d->vec + d->o_sin + 256, 256);
    float *h1T = d->vec + d->o_h1, *h10 = h1T + dim;
    iris_dev_linear_f32(h1T, d->vec_b16 + d->o_sin, d->time_lin1, 1, 256, dim);
    iris_dev_silu(h1T, dim);
    iris_dev_f32_to_bf16(d->vec_b16 + d->o_h1, h1T, dim);
    float *tembT = d->vec + d->o_temb, *temb0 = tembT + dim;
    iris_dev_linear_f32(tembT, d->vec_b16 + d->o_h1, d->time_lin2, 1, dim, dim);
    iris_dev_linear_f32(h10, d->vec_b16 + d->o_sin + 256, d->time_lin1, 1, 256, dim);
    iris_dev_silu(h10, dim);
    iris_dev_f32_to_bf16(d->vec_b16 + d->o_h1 + dim, h10, dim);
    iris_dev_linear_f32(temb0, d->vec_b16 + d->o_h1 + dim, d->time_lin2, 1, dim, dim);

    /* Modulation rows (shared across all blocks) and output scales. */
    iris_dev_f32_to_bf16(d->vec_b16 + d->o_act, tembT, dim);
    iris_dev_f32_to_bf16(d->vec_b16 + d->o_act + dim, temb0, dim);
    float *modT = d->vec + d->o_mod, *mod0 = modT + 4 * dim;
    iris_dev_linear_f32(modT, d->vec_b16 + d->o_act, d->modulation, 1, dim, 4 * dim);
    iris_dev_linear_f32(mod0, d->vec_b16 + d->o_act + dim, d->modulation, 1, dim, 4 * dim);
    iris_dev_linear_f32(d->vec + d->o_sout, d->vec_b16 + d->o_act, d->norm_out_lin, 1, dim, dim);
    iris_dev_linear_f32(d->vec + d->o_sout + dim, d->vec_b16 + d->o_act + dim, d->norm_out_lin, 1, dim, dim);

    /* tanh of the gate rows (iris_dev_residual has no tanh): the device API
     * lacks tanh, so download the two modulation rows, tanh their gate parts
     * on the host, and upload the four gate vectors. */
    {
        static const float zero[4] = {0,0,0,0};
        iris_dev_upload(d->vec + d->o_tg,        zero, 0);
        iris_dev_sync();   /* ensure the modulation GEMMs landed before read */
        iris_dev_download(modT, modT, 4 * dim * sizeof(float));
        iris_dev_download(mod0, mod0, 4 * dim * sizeof(float));
        float *g1T = d->vec + d->o_tg, *g2T = g1T + dim, *g10 = g2T + dim, *g20 = g10 + dim;
        for (int j = 0; j < dim; j++) { g1T[j] = tanhf(modT[dim + j]);     g2T[j] = tanhf(modT[3 * dim + j]); }
        for (int j = 0; j < dim; j++) { g10[j] = tanhf(mod0[dim + j]);     g20[j] = tanhf(mod0[3 * dim + j]); }
    }

    /* RoPE cos/sin (device layout: [seq, head_dim/2], consecutive pairs). */
    float *qk_c = malloc((size_t)N * hd * sizeof(float));
    float *qk_s = malloc((size_t)N * hd * sizeof(float));
    int *kv = malloc(N * sizeof(int));
    for (int i = 0; i < s_txt; i++) {
        kv[i] = txt_mask ? txt_mask[i] : 1;
        qw_token_rope(tf, i, i, i, qk_c + (size_t)i * hd, qk_s + (size_t)i * hd);
    }
    int hh = img_h - img_h / 2, ww = img_w - img_w / 2;
    for (int r = 0; r < img_h; r++)
        for (int c = 0; c < img_w; c++) {
            int i = s_txt + r * img_w + c;
            kv[i] = 1;
            qw_token_rope(tf, s_txt, r - hh, c - ww, qk_c + (size_t)i * hd, qk_s + (size_t)i * hd);
        }
    {
        float *rc = malloc((size_t)N * half * sizeof(float));
        float *rs = malloc((size_t)N * half * sizeof(float));
        for (int i = 0; i < N; i++)
            for (int p = 0; p < half; p++) {
                rc[(size_t)i * half + p] = qk_c[(size_t)i * hd + 2 * p];
                rs[(size_t)i * half + p] = qk_s[(size_t)i * hd + 2 * p];
            }
        iris_dev_upload(d->rope_c, rc, (size_t)N * half * sizeof(float));
        iris_dev_upload(d->rope_s, rs, (size_t)N * half * sizeof(float));
        free(rc); free(rs);
    }
    iris_dev_upload(d->mask, kv, (size_t)N * sizeof(int));
    int *kv_host = kv;   /* kept: host text attention needs the validity flags */
    free(qk_c); free(qk_s);

    float scale = 1.0f / sqrtf((float)hd);
    /* Blocks. */
    for (int b = 0; b < tf->n_layers; b++) {
        qw_dev_block_t *bl = &d->blocks[b];
        for (int seg = 0; seg < 2; seg++) {
            int start = seg ? s_txt : 0, len = seg ? n_img : s_txt;
            if (len <= 0) continue;
            const float *m = seg ? modT : mod0;
            iris_dev_layer_norm_mod(d->norm + (size_t)start * dim, d->x + (size_t)start * dim,
                                    m, m + dim, len, dim, tf->eps);
        }
        iris_dev_linear_bf16(d->qkv, d->norm, bl->to_q, N, dim, dim);
        iris_dev_linear_bf16(d->qkv + dim, d->norm, bl->to_k, N, dim, dim);
        iris_dev_linear_bf16(d->qkv + 2 * dim, d->norm, bl->to_v, N, dim, dim);
        iris_dev_head_norm_rope(d->qkv, 3 * dim, bl->norm_q, d->rope_c, d->rope_s, N, heads, hd, tf->eps, 0);
        iris_dev_head_norm_rope(d->qkv + dim, 3 * dim, bl->norm_k, d->rope_c, d->rope_s, N, heads, hd, tf->eps, 0);
        qw_attn_host(tf, d->qkv, s_txt, n_img, kv_host, d->att);
        if (n_img > 0)
            iris_dev_attention(d->att + (size_t)s_txt * dim, dim,
                               d->qkv + (size_t)s_txt * dim, d->qkv + dim,
                               d->qkv + 2 * dim, 3 * dim,
                               n_img, N, heads, heads, hd, scale, 0, NULL);
        iris_dev_linear_bf16(d->gate, d->norm, bl->mlp_gate, N, dim, mlp);
        iris_dev_linear_bf16(d->ffn, d->norm, bl->mlp_proj, N, dim, mlp);
        iris_dev_swiglu(d->ffn, mlp, d->gate, d->ffn, mlp, N, mlp);
        iris_dev_linear_f32(d->o1, d->att, bl->to_out, N, dim, dim);
        iris_dev_linear_f32(d->o2, d->ffn, bl->mlp_out, N, mlp, dim);
        /* o1/o2 are GPU-written; flush before the host reads them to sum. */
        iris_dev_sync();
        for (int seg = 0; seg < 2; seg++) {
            int start = seg ? s_txt : 0, len = seg ? n_img : s_txt;
            if (len <= 0) continue;
            float *o = d->o1 + (size_t)start * dim, *o2 = d->o2 + (size_t)start * dim;
            for (size_t e = 0; e < (size_t)len * dim; e++) o[e] += o2[e];
            iris_dev_residual(d->x + (size_t)start * dim, o, d->vec + d->o_tg + (seg ? 2 * dim : 0), len, dim);
        }
        iris_dev_sync();
    }

    /* norm_out: LayerNorm (no shift/scale) then * (1 + soutT), then proj_out
     * on the image rows. The device API faults on a NULL scale/shift LN, so do
     * the LN and per-row scale on the host and re-stash bf16. The upload goes
     * through d->o1 (N*dim f32): d->in_f32 is only max(s_txt*ctx,
     * n_img*in_ch) elements and would overflow for n_img*dim. */
    {
        float *nf = malloc((size_t)n_img * dim * sizeof(float));
        /* d->x is f32 (not bf16): download it directly, no bf16 widening. */
        iris_dev_download(nf, d->x + (size_t)s_txt * dim,
                          (size_t)n_img * dim * sizeof(float));
        qw_layernorm(nf, n_img, dim, tf->eps);
        const float *sc = d->vec + d->o_sout;   /* soutT: image-row scale */
        for (size_t i = 0; i < (size_t)n_img * dim; i++) nf[i] *= 1.0f + sc[i % dim];
        iris_dev_upload(d->o1, nf, (size_t)n_img * dim * sizeof(float));
        free(nf);
        iris_dev_f32_to_bf16(d->att, d->o1, (size_t)n_img * dim);
        iris_dev_linear_f32(d->in_f32, d->att, d->proj_out, n_img, dim, tf->in_ch);
    }
    iris_dev_sync();
    iris_dev_download(v, d->in_f32, (size_t)n_img * tf->in_ch * sizeof(float));
    return 1;
}
#endif /* IRIS_DEV */

void qw_transformer_forward(qw_transformer_t *tf,
                            const float *z, const float *txt,
                            const int *txt_mask, int s_txt,
                            int img_h, int img_w,
                            float sigma, float *v)
{
#ifdef IRIS_DEV
    if (tf->dev) {
        if (qw_forward_dev(tf, z, txt, txt_mask, s_txt, img_h, img_w, sigma, v))
            return;
        /* device forward failed: fall back to the CPU path */
    }
#endif
    int dim = tf->dim, hd = tf->head_dim;
    int n_img = img_h * img_w;
    int N = s_txt + n_img;
    float t = 1000.0f * sigma;

    /* Joint sequence: [text, target]. */
    float *x = malloc((size_t)N * dim * sizeof(float));
    float *enc = x;
    memcpy(enc, txt, (size_t)s_txt * tf->ctx_dim * sizeof(float));
    if (s_txt > 0) {
        qw_rmsnorm(enc, s_txt, tf->ctx_dim, tf->txt_norm_w, 1, tf->eps);
        float *p = malloc((size_t)s_txt * dim * sizeof(float));
        qw_gemm(enc, tf->txt_in, p, s_txt, tf->ctx_dim, dim);
        qw_gelu_tanh(p, s_txt * dim);
        qw_gemm(p, tf->txt_out, enc, s_txt, dim, dim);
        free(p);
    }
    qw_gemm(z, tf->img_in, enc + (size_t)s_txt * dim, n_img, tf->in_ch, dim);

    /* Timestep embeddings for the sampled sigma and for t = 0. t is already
     * the 1000-scaled timestep (sigma * 1000), the exact sinusoidal input the
     * reference uses. */
    float embT[256], emb0[256];
    qw_time_embed(embT, t);
    for (int j = 0; j < 128; j++) { emb0[j] = 1.0f; emb0[128 + j] = 0.0f; }
    float *tembT = malloc(dim * sizeof(float));
    float *temb0 = malloc(dim * sizeof(float));
    float *h1 = malloc(dim * sizeof(float));
    qw_gemm(embT, tf->time_lin1, h1, 1, 256, dim);
    qw_silu(h1, dim);
    memcpy(tembT, h1, dim * sizeof(float));
    qw_gemm(h1, tf->time_lin2, tembT, 1, dim, dim);
    qw_gemm(emb0, tf->time_lin1, h1, 1, 256, dim);
    qw_silu(h1, dim);
    memcpy(temb0, h1, dim * sizeof(float));
    qw_gemm(h1, tf->time_lin2, temb0, 1, dim, dim);
    free(h1);

    /* Shared modulation and final-norm scale: nn.Sequential(SiLU, Linear)
     * and linear(silu(temb)) respectively, on each of the two rows. */
    float *modT = malloc(4 * dim * sizeof(float));
    float *mod0 = malloc(4 * dim * sizeof(float));
    float *soutT = malloc(dim * sizeof(float));
    float *sout0 = malloc(dim * sizeof(float));
    float *ts = malloc(dim * sizeof(float));
    memcpy(ts, tembT, dim * sizeof(float)); qw_silu(ts, dim);
    qw_gemm(ts, tf->modulation, modT, 1, dim, 4 * dim);
    qw_gemm(ts, tf->norm_out_lin, soutT, 1, dim, dim);
    memcpy(ts, temb0, dim * sizeof(float)); qw_silu(ts, dim);
    qw_gemm(ts, tf->modulation, mod0, 1, dim, 4 * dim);
    qw_gemm(ts, tf->norm_out_lin, sout0, 1, dim, dim);
    free(ts);

    /* Per-token token classification and RoPE tables. */
    int *is_target = malloc(N * sizeof(int));
    int *key_valid = malloc(N * sizeof(int));
    float *qk_c = malloc((size_t)N * hd * sizeof(float));
    float *qk_s = malloc((size_t)N * hd * sizeof(float));
    for (int i = 0; i < s_txt; i++) {
        is_target[i] = 0;
        key_valid[i] = txt_mask ? txt_mask[i] : 1;
        int pos = i; /* all three axes advance with the text cursor */
        qw_token_rope(tf, pos, pos, pos,
                      qk_c + (size_t)i * hd, qk_s + (size_t)i * hd);
    }
    int hh = img_h - img_h / 2;   /* center offset, e.g. 16 for 32 */
    int ww = img_w - img_w / 2;
    for (int r = 0; r < img_h; r++) {
        for (int c = 0; c < img_w; c++) {
            int i = s_txt + r * img_w + c;
            is_target[i] = 1;
            key_valid[i] = 1;
            /* Frame frozen at the position reached by the preceding text; */
            /* height/width centred on zero.                            */
            qw_token_rope(tf, s_txt, r - hh, c - ww,
                          qk_c + (size_t)i * hd, qk_s + (size_t)i * hd);
        }
    }

    /* Blocks */
    float *xm = malloc((size_t)N * dim * sizeof(float));
    for (int b = 0; b < tf->n_layers; b++) {
        qw_block(tf, b, x, xm, N, modT, mod0, is_target, qk_c, qk_s,
                 s_txt, N, key_valid);
    }
    free(xm);

    /* norm_out: LN * (1 + scale), scale from the token's modulation row. */
    qw_layernorm(x, N, dim, tf->eps);
    for (int i = 0; i < N; i++) {
        const float *sc = is_target[i] ? soutT : sout0;
        float *row = x + (size_t)i * dim;
        for (int j = 0; j < dim; j++) row[j] *= 1.0f + sc[j];
    }

    /* proj_out on the whole joint sequence; the target block's rows are the
     * model's velocity prediction. */
    float *out = malloc((size_t)N * tf->in_ch * sizeof(float));
    qw_gemm(x, tf->proj_out, out, N, dim, tf->in_ch);
    memcpy(v, out + (size_t)s_txt * tf->in_ch, (size_t)n_img * tf->in_ch * sizeof(float));

    free(out); free(x);
    free(modT); free(mod0); free(tembT); free(temb0);
    free(soutT); free(sout0);
    free(is_target); free(key_valid); free(qk_c); free(qk_s);
}

/* ========================================================================
 * Euler sampling loop
 * ======================================================================== */

/*
 * Denoise z in place over the given sigma schedule (length num_steps + 1,
 * ending at 0). Returns z.
 */
float *iris_sample_euler_qwen21(qw_transformer_t *tf,
                                float *z, int img_h, int img_w,
                                const float *txt, int s_txt,
                                const int *txt_mask,
                                const float *schedule, int num_steps,
                                void (*progress)(int step, int total))
{
    int n = img_h * img_w;
    float *vel = malloc((size_t)n * tf->in_ch * sizeof(float));
    for (int i = 0; i < num_steps; i++) {
        qw_transformer_forward(tf, z, txt, txt_mask, s_txt, img_h, img_w,
                               schedule[i], vel);
        float dsig = schedule[i + 1] - schedule[i];
        for (int j = 0; j < n * tf->in_ch; j++) z[j] += dsig * vel[j];
        if (progress) progress(i + 1, num_steps);
    }
    free(vel);
    return z;
}
