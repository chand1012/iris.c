/*
 * Qwen3-VL vision tower (Qwen-Image 2.1 text encoder's model.visual.*)
 *
 * 27 blocks, hidden 1152, 16 heads x 72 dim, GELU(tanh) FFN (4304),
 * full bidirectional attention per image (cu_seqlens = one span per image).
 * Patch embed is a Conv3d whose input rows already match the kernel, so it
 * is a plain GEMM. Learned 48x48 pos_embed table is resampled bilinearly
 * (align_corners=True) into each image's grid; patches are in merge-block
 * order (block_row, block_col, in_row, in_col), matching the processor.
 * Axial 2D rope over (h,w): 18 freqs per axis, head layout [h|w|h|w] with
 * rotate_half pairing (d, d+36). Merger: LN(1152) -> 4-merge -> fc1(4608)
 * -> exact GELU -> fc2(4096). Deepstack mergers (blocks 8/16/24) are the
 * same but LN over the merged 4608 rows.
 *
 * Reference: transformers/models/qwen3_vl/modeling_qwen3_vl.py
 */

#include "iris_vision_qwen3vl.h"
#include "iris_safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <math.h>

#ifdef USE_BLAS
#if defined(IRIS_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(USE_OPENBLAS)
#include <cblas.h>
#endif
#endif

#define QWV_MAX_SHARDS 16
#define QWV_MAX_DEEPSTACK 4

/* bf16 -> f32: a bf16 value is the upper 16 bits of an f32. */
static float qwv_bf2f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f; memcpy(&f, &u, sizeof f); return f;
}

/* f32 -> bf16 with round-to-nearest-even on the IEEE bit pattern. */
static uint16_t qwv_f2bf(float f) {
    uint32_t u; memcpy(&u, &f, sizeof u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* Bounded f32 scratch for widening bf16 weights (one matrix at a time). */
static float *qwv_wscratch = NULL;
static size_t qwv_wscratch_cap = 0;
static float *qwv_bscratch = NULL;   /* bias widening scratch */
static size_t qwv_bscratch_cap = 0;
static float *qwv_bscratch2 = NULL;  /* second bias scratch (LN w+b pair) */
static size_t qwv_bscratch2_cap = 0;

static float *qwv_wide(const uint16_t *w, size_t nel) {
    if (nel > qwv_wscratch_cap) {
        free(qwv_wscratch);
        qwv_wscratch = malloc(nel * sizeof(float));
        qwv_wscratch_cap = qwv_wscratch ? nel : 0;
    }
    float *o = qwv_wscratch;
    for (size_t i = 0; i < nel; i++) o[i] = qwv_bf2f(w[i]);
    return o;
}

/* Bias/vector widening into a separate scratch (kept distinct from the
 * weight scratch so a GEMM's widened W survives the bias add). */
static float *qwv_wide_bias(const uint16_t *v, size_t n) {
    if (n > qwv_bscratch_cap) {
        free(qwv_bscratch);
        qwv_bscratch = malloc(n * sizeof(float));
        qwv_bscratch_cap = qwv_bscratch ? n : 0;
    }
    float *o = qwv_bscratch;
    for (size_t i = 0; i < n; i++) o[i] = qwv_bf2f(v[i]);
    return o;
}

/* Row-major GEMM: out[n,o] = in[n,k] x W[o,k]^T (+bias if given). */
static void qwv_gemm(const float *in, const uint16_t *W, const uint16_t *bias,
                     float *out, int n, int k, int o) {
    float *Wf = qwv_wide(W, (size_t)k * o);
#ifdef USE_BLAS
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
    if (bias) {
        float *b = qwv_wide_bias(bias, o);   /* after the GEMM: Wf is done */
        for (int i = 0; i < n; i++)
            for (int j = 0; j < o; j++) out[(size_t)i * o + j] += b[j];
    }
}

/* LayerNorm with affine params (bf16, widened into two distinct scratches
 * so the w and b pointers cannot alias), eps 1e-6 (torch semantics). */
static void qwv_ln(const float *x, float *out, int n, int d,
                   const uint16_t *bw, const uint16_t *bb) {
    if (d > qwv_bscratch_cap) {
        free(qwv_bscratch);
        qwv_bscratch = malloc(d * sizeof(float));
        qwv_bscratch_cap = qwv_bscratch ? d : 0;
    }
    if (d > qwv_bscratch2_cap) {
        free(qwv_bscratch2);
        qwv_bscratch2 = malloc(d * sizeof(float));
        qwv_bscratch2_cap = qwv_bscratch2 ? d : 0;
    }
    float *w = qwv_bscratch, *b = qwv_bscratch2;
    for (int j = 0; j < d; j++) { w[j] = qwv_bf2f(bw[j]); b[j] = qwv_bf2f(bb[j]); }
    for (int i = 0; i < n; i++) {
        const float *row = x + (size_t)i * d;
        float *orow = out + (size_t)i * d;
        float mean = 0.0f;
        for (int j = 0; j < d; j++) mean += row[j];
        mean /= d;
        float var = 0.0f;
        for (int j = 0; j < d; j++) { float t = row[j] - mean; var += t * t; }
        float rstd = 1.0f / sqrtf(var / d + 1e-6f);
        for (int j = 0; j < d; j++) orow[j] = (row[j] - mean) * rstd * w[j] + b[j];
    }
}

static void qwv_gelu_tanh(float *x, size_t n) {
    const float c = 0.7978845608028654f;
    for (size_t i = 0; i < n; i++) {
        float t = x[i];
        x[i] = 0.5f * t * (1.0f + tanhf(c * (t + 0.044715f * t * t * t)));
    }
}

static void qwv_gelu_exact(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float t = x[i];
        x[i] = 0.5f * t * (1.0f + erff(t * 0.7071067811865476f));
    }
}

/* ========================================================================
 * Model state
 * ======================================================================== */

typedef struct {
    uint16_t *ln1w, *ln1b, *ln2w, *ln2b;
    uint16_t *qkvw, *qkvb, *pw, *pb;
    uint16_t *f1w, *f1b, *f2w, *f2b;
} qwv_block_w;

typedef struct {
    uint16_t *nw, *nb;         /* LN weights (1152, or 4608 postshuffle) */
    uint16_t *f1w, *f1b, *f2w, *f2b;
} qwv_merger_w;

struct qwv_model {
    safetensors_file_t *sf[QWV_MAX_SHARDS];
    int n_sf;

    int depth, hidden, heads, head_dim, mlp;
    int patch, tpatch, merge, mlp_m;   /* mlp_m = merge^2 */
    int in_ch, out_h;
    int n_grid;                        /* sqrt(num_position_embeddings) */
    float theta;
    int ds_idx[QWV_MAX_DEEPSTACK];     /* block numbers with deepstack mergers */
    int n_ds;

    uint16_t *pe_w, *pe_b;             /* patch embed conv-as-GEMM */
    uint16_t *pos_tab;                 /* [n_grid^2, hidden] */
    qwv_block_w *blocks;
    qwv_merger_w merger_w, ds_w[QWV_MAX_DEEPSTACK];

    /* forward scratch */
    float *x, *nb, *qkv, *att, *fc1, *cos, *sin;
    float *sc, *qc, *kc, *vc;   /* attention score + per-head copies */
};

static uint16_t *find_t(qwv_model_t *m, const char *name) {
    const safetensor_t *t;
    for (int i = 0; i < m->n_sf; i++)
        if ((t = safetensors_find(m->sf[i], name)))
            return safetensors_get_bf16_direct(m->sf[i], t);
    return NULL;
}

static int find_load(qwv_model_t *m, uint16_t **dst, const char *fmt, ...) {
    char name[128];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name, sizeof name, fmt, ap);
    va_end(ap);
    *dst = find_t(m, name);
    return *dst ? 0 : -1;
}

/* Config: read vision_config object out of config.json (flat strstr parse). */
static double json_num(const char *block, const char *key, double dflt) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\"", key);
    const char *p = strstr(block, pat);
    if (!p) return dflt;
    p += strlen(pat);
    while (*p && *p != ':') p++;
    if (*p != ':') return dflt;
    return atof(p + 1);
}

static int qwv_read_config(qwv_model_t *m, const char *dir) {
    char path[1024];
    snprintf(path, sizeof path, "%s/config.json", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "qwv: missing %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (!buf) { fclose(f); return -1; }
    size_t rd = fread(buf, 1, sz, f);
    buf[rd] = '\0';
    fclose(f);

    const char *vc = strstr(buf, "\"vision_config\"");
    if (!vc) { free(buf); fprintf(stderr, "qwv: no vision_config in %s\n", path); return -1; }
    vc = strchr(vc, '{');
    if (!vc) { free(buf); return -1; }
    /* end of the vision_config object: matching close brace */
    int depth = 0;
    const char *q = vc;
    while (*q) {
        if (*q == '{') depth++;
        else if (*q == '}' && --depth == 0) break;
        q++;
    }
    size_t blen = (size_t)(q - vc + 1);
    char *block = malloc(blen + 1);
    memcpy(block, vc, blen);
    block[blen] = '\0';
    free(buf);

    m->depth = (int)json_num(block, "depth", 0);
    m->hidden = (int)json_num(block, "hidden_size", 0);
    m->heads = (int)json_num(block, "num_heads", 0);
    m->mlp = (int)json_num(block, "intermediate_size", 0);
    m->patch = (int)json_num(block, "patch_size", 16);
    m->tpatch = (int)json_num(block, "temporal_patch_size", 2);
    m->merge = (int)json_num(block, "spatial_merge_size", 2);
    m->in_ch = (int)json_num(block, "in_channels", 3);
    m->out_h = (int)json_num(block, "out_hidden_size", 0);
    m->n_grid = (int)sqrt(json_num(block, "num_position_embeddings", 0));
    m->theta = (float)json_num(block, "rope_theta", 10000.0);

    /* deepstack_visual_indexes: [8, 16, 24] */
    const char *p = strstr(block, "\"deepstack_visual_indexes\"");
    m->n_ds = 0;
    if (p && (p = strchr(p, '['))) {
        p++;
        while (*p && *p != ']' && m->n_ds < QWV_MAX_DEEPSTACK) {
            if (*p >= '0' && *p <= '9') {
                m->ds_idx[m->n_ds++] = (int)strtol(p, (char **)&p, 10);
            } else p++;
        }
    }
    free(block);
    m->head_dim = m->hidden / m->heads;
    m->mlp_m = m->merge * m->merge;
    if (m->depth <= 0 || m->hidden <= 0 || m->heads <= 0 || m->out_h <= 0) return -1;
    return 0;
}

qwv_model_t *qwv_load(const char *dir) {
    qwv_model_t *m = calloc(1, sizeof *m);
    if (!m) return NULL;
    if (qwv_read_config(m, dir) != 0) { free(m); return NULL; }

    /* open shards via the index file */
    char path[1024];
    snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "qwv: missing %s\n", path); free(m); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(sz + 1);
    if (!buf) { fclose(f); free(m); return NULL; }
    size_t rd = fread(buf, 1, sz, f);
    buf[rd] = '\0';
    fclose(f);
    char seen[QWV_MAX_SHARDS][128];
    int n = 0;
    for (char *p = buf; (p = strstr(p, ".safetensors"));) {
        char *s = p - 1;
        while (s > buf && *s != '"' && *s != ',' && *s != ':') s--;
        s++;
        int len = (int)(p - s) + (int)strlen(".safetensors");
        if (len < 128 && n < QWV_MAX_SHARDS) {
            memcpy(seen[n], s, len);
            seen[n][len] = '\0';
            int dup = 0;
            for (int i = 0; i < n; i++) if (!strcmp(seen[i], seen[n])) { dup = 1; break; }
            if (!dup) n++;
        }
        p += strlen(".safetensors");
    }
    free(buf);
    if (!n) { free(m); return NULL; }
    for (int i = 0; i < n; i++) {
        snprintf(path, sizeof path, "%s/%s", dir, seen[i]);
        m->sf[i] = safetensors_open(path);
        if (!m->sf[i]) {
            fprintf(stderr, "qwv: failed to open %s\n", path);
            for (int j = 0; j < i; j++) safetensors_close(m->sf[j]);
            free(m);
            return NULL;
        }
    }
    m->n_sf = n;

    /* resolve weight pointers (mmap-backed bf16) */
    if (find_load(m, &m->pe_w, "model.visual.patch_embed.proj.weight") ||
        find_load(m, &m->pe_b, "model.visual.patch_embed.proj.bias") ||
        find_load(m, &m->pos_tab, "model.visual.pos_embed.weight")) {
        fprintf(stderr, "qwv: missing patch/pos weights\n");
        qwv_free(m);
        return NULL;
    }
    m->blocks = calloc(m->depth, sizeof *m->blocks);
    if (!m->blocks) { qwv_free(m); return NULL; }
    for (int i = 0; i < m->depth; i++) {
        qwv_block_w *b = &m->blocks[i];
        if (find_load(m, &b->ln1w, "model.visual.blocks.%d.norm1.weight", i) ||
            find_load(m, &b->ln1b, "model.visual.blocks.%d.norm1.bias", i) ||
            find_load(m, &b->ln2w, "model.visual.blocks.%d.norm2.weight", i) ||
            find_load(m, &b->ln2b, "model.visual.blocks.%d.norm2.bias", i) ||
            find_load(m, &b->qkvw, "model.visual.blocks.%d.attn.qkv.weight", i) ||
            find_load(m, &b->qkvb, "model.visual.blocks.%d.attn.qkv.bias", i) ||
            find_load(m, &b->pw,  "model.visual.blocks.%d.attn.proj.weight", i) ||
            find_load(m, &b->pb,  "model.visual.blocks.%d.attn.proj.bias", i) ||
            find_load(m, &b->f1w, "model.visual.blocks.%d.mlp.linear_fc1.weight", i) ||
            find_load(m, &b->f1b, "model.visual.blocks.%d.mlp.linear_fc1.bias", i) ||
            find_load(m, &b->f2w, "model.visual.blocks.%d.mlp.linear_fc2.weight", i) ||
            find_load(m, &b->f2b, "model.visual.blocks.%d.mlp.linear_fc2.bias", i)) {
            fprintf(stderr, "qwv: missing vision block %d weights\n", i);
            qwv_free(m);
            return NULL;
        }
    }
    qwv_merger_w *mg = &m->merger_w;
    if (find_load(m, &mg->nw, "model.visual.merger.norm.weight") ||
        find_load(m, &mg->nb, "model.visual.merger.norm.bias") ||
        find_load(m, &mg->f1w, "model.visual.merger.linear_fc1.weight") ||
        find_load(m, &mg->f1b, "model.visual.merger.linear_fc1.bias") ||
        find_load(m, &mg->f2w, "model.visual.merger.linear_fc2.weight") ||
        find_load(m, &mg->f2b, "model.visual.merger.linear_fc2.bias")) {
        fprintf(stderr, "qwv: missing merger weights\n");
        qwv_free(m);
        return NULL;
    }
    for (int d = 0; d < m->n_ds; d++) {
        qwv_merger_w *dw = &m->ds_w[d];
        if (find_load(m, &dw->nw, "model.visual.deepstack_merger_list.%d.norm.weight", d) ||
            find_load(m, &dw->nb, "model.visual.deepstack_merger_list.%d.norm.bias", d) ||
            find_load(m, &dw->f1w, "model.visual.deepstack_merger_list.%d.linear_fc1.weight", d) ||
            find_load(m, &dw->f1b, "model.visual.deepstack_merger_list.%d.linear_fc1.bias", d) ||
            find_load(m, &dw->f2w, "model.visual.deepstack_merger_list.%d.linear_fc2.weight", d) ||
            find_load(m, &dw->f2b, "model.visual.deepstack_merger_list.%d.linear_fc2.bias", d)) {
            fprintf(stderr, "qwv: missing deepstack merger %d weights\n", d);
            qwv_free(m);
            return NULL;
        }
    }
    return m;
}

void qwv_free(qwv_model_t *m) {
    if (!m) return;
    for (int i = 0; i < m->n_sf; i++) safetensors_close(m->sf[i]);
    free(m->blocks);
    free(m->x); free(m->nb); free(m->qkv); free(m->att);
    free(m->fc1); free(m->cos); free(m->sin);
    free(m->sc); free(m->qc); free(m->kc); free(m->vc);
    free(m);
}

/* ========================================================================
 * Forward
 * ======================================================================== */

/* Merger (both flavors): norm -> fc1 -> GELU -> fc2.
 * postshuffle: LN over merged mlp_m*hidden rows; else LN over hidden rows
 * before the reshape. Output [n/mlp_m, out_h] into `out`. */
static void qwv_merge(qwv_model_t *m, const qwv_merger_w *w, const float *in,
                      int n, int postshuffle, float *out) {
    int dm = m->hidden * m->mlp_m;      /* fc1 in/out (4608) */
    int rows = n / m->mlp_m;
    memcpy(m->nb, in, (size_t)n * m->hidden * sizeof(float));
    qwv_ln(m->nb, m->nb, postshuffle ? rows : n,
           postshuffle ? dm : m->hidden, w->nw, w->nb);
    qwv_gemm(m->nb, w->f1w, w->f1b, m->fc1, rows, dm, dm);
    qwv_gelu_exact(m->fc1, (size_t)rows * dm);
    qwv_gemm(m->fc1, w->f2w, w->f2b, out, rows, dm, m->out_h);
}

int qwv_forward(qwv_model_t *m, int t, int h, int w, const float *pixels,
                float *merged, float *deepstack_out) {
    int hd = m->head_dim, hid = m->hidden;
    int n = t * h * w;
    if ((h % m->merge) || (w % m->merge)) return -1;

    /* scratch sized to this call */
    size_t nx = (size_t)n * hid;
    free(m->x); m->x = malloc(nx * sizeof(float));
    free(m->nb); m->nb = malloc(nx * sizeof(float));
    free(m->qkv); m->qkv = malloc((size_t)n * 3 * hid * sizeof(float));
    free(m->att); m->att = malloc(nx * sizeof(float));
    free(m->fc1); m->fc1 = malloc((size_t)n * m->mlp * sizeof(float));
    free(m->cos); m->cos = malloc((size_t)n * hd * sizeof(float));
    free(m->sin); m->sin = malloc((size_t)n * hd * sizeof(float));
    free(m->sc); m->sc = malloc((size_t)n * n * sizeof(float));
    free(m->qc); m->qc = malloc((size_t)n * hd * sizeof(float));
    free(m->kc); m->kc = malloc((size_t)n * hd * sizeof(float));
    free(m->vc); m->vc = malloc((size_t)n * hd * sizeof(float));
    if (!m->x || !m->nb || !m->qkv || !m->att || !m->fc1 || !m->cos || !m->sin ||
        !m->sc || !m->qc || !m->kc || !m->vc)
        return -1;

    /* patch embed: rows already match the conv kernel -> GEMM */
    int pin = m->in_ch * m->tpatch * m->patch * m->patch;
    qwv_gemm(pixels, m->pe_w, m->pe_b, m->x, n, pin, hid);

    /* pos embed: bilinear resample of the n_grid^2 table (align_corners=True),
     * merge-block patch order: within -> (row, col). */
    {
        int side = m->n_grid;
        int blocks_w = w / m->merge;
        for (int p = 0; p < n; p++) {
            int within = p % (h * w);
            int in_col = within % m->merge;
            int in_row = (within / m->merge) % m->merge;
            int block_col = (within / (m->merge * m->merge)) % blocks_w;
            int block_row = within / (m->merge * m->merge * blocks_w);
            int row = block_row * m->merge + in_row;
            int col = block_col * m->merge + in_col;
            float src_h = (h > 1) ? row * (float)(side - 1) / (float)(h - 1) : 0.0f;
            float src_w = (w > 1) ? col * (float)(side - 1) / (float)(w - 1) : 0.0f;
            int fh = (int)floorf(src_h), fw = (int)floorf(src_w);
            float wh = 1.0f - (src_h - fh), ww = 1.0f - (src_w - fw);
            int h0 = fh < 0 ? 0 : (fh > side - 1 ? side - 1 : fh);
            int h1 = fh + 1 < 0 ? 0 : (fh + 1 > side - 1 ? side - 1 : fh + 1);
            int w0 = fw < 0 ? 0 : (fw > side - 1 ? side - 1 : fw);
            int w1 = fw + 1 < 0 ? 0 : (fw + 1 > side - 1 ? side - 1 : fw + 1);
            float wa = wh * ww, wb = wh * (1 - ww), wc = (1 - wh) * ww, wd = (1 - wh) * (1 - ww);
            const uint16_t *ta = m->pos_tab + (size_t)(h0 * side + w0) * hid;
            const uint16_t *tb = m->pos_tab + (size_t)(h0 * side + w1) * hid;
            const uint16_t *tc = m->pos_tab + (size_t)(h1 * side + w0) * hid;
            const uint16_t *td = m->pos_tab + (size_t)(h1 * side + w1) * hid;
            float *xo = m->x + (size_t)p * hid;
            for (int j = 0; j < hid; j++)
                xo[j] += wa * qwv_bf2f(ta[j]) + wb * qwv_bf2f(tb[j]) +
                         wc * qwv_bf2f(tc[j]) + wd * qwv_bf2f(td[j]);
        }
    }

    /* axial rope: per patch (hpos, wpos) in merge-block order; head layout
     * [h_freqs | w_freqs | h_freqs | w_freqs], rotate_half pairs (d, d+hd/2). */
    {
        int half = hd / 2, nf = half / 2;
        float *invf = malloc(nf * sizeof(float));
        for (int i = 0; i < nf; i++) {
            /* the reference model holds rope buffers in bf16: its inv_freq
             * is bf16-rounded before being widened to f32 for the angles */
            float f0 = 1.0f / powf(m->theta, (float)(2 * i) / (float)half);
            invf[i] = qwv_bf2f(qwv_f2bf(f0));
        }
        int blocks_w = w / m->merge;
        for (int p = 0; p < n; p++) {
            int within = p % (h * w);
            int in_col = within % m->merge;
            int in_row = (within / m->merge) % m->merge;
            int block_col = (within / (m->merge * m->merge)) % blocks_w;
            int block_row = within / (m->merge * m->merge * blocks_w);
            float hp = (float)(block_row * m->merge + in_row);
            float wp = (float)(block_col * m->merge + in_col);
            float *c = m->cos + (size_t)p * hd, *s = m->sin + (size_t)p * hd;
            for (int i = 0; i < nf; i++) {
                float ah = hp * invf[i], aw = wp * invf[i];
                c[i] = cosf(ah); s[i] = sinf(ah);
                c[half / 2 + i] = cosf(aw); s[half / 2 + i] = sinf(aw);
                c[half + i] = c[i]; s[half + i] = s[i];
                c[half + half / 2 + i] = c[half / 2 + i];
                s[half + half / 2 + i] = s[half / 2 + i];
            }
        }
        free(invf);
    }

    /* blocks: full attention within each image (cu_seqlens = image spans) */
    for (int l = 0; l < m->depth; l++) {
        qwv_block_w *b = &m->blocks[l];
        qwv_ln(m->x, m->nb, n, hid, b->ln1w, b->ln1b);
        qwv_gemm(m->nb, b->qkvw, b->qkvb, m->qkv, n, hid, 3 * hid);
        /* rope on q,k (in place inside qkv rows) */
        for (int p = 0; p < n; p++) {
            const float *c = m->cos + (size_t)p * hd, *s = m->sin + (size_t)p * hd;
            for (int hq = 0; hq < m->heads; hq++) {
                float *q = m->qkv + (size_t)p * 3 * hid + (size_t)hq * hd;
                float *k = q + hid;
                for (int d = 0; d < hd / 2; d++) {
                    float qa = q[d], qb = q[d + hd / 2];
                    q[d] = qa * c[d] - qb * s[d];
                    q[d + hd / 2] = qb * c[d + hd / 2] + qa * s[d + hd / 2];
                    float ka = k[d], kb = k[d + hd / 2];
                    k[d] = ka * c[d] - kb * s[d];
                    k[d + hd / 2] = kb * c[d + hd / 2] + ka * s[d + hd / 2];
                }
            }
        }
        /* Full bidirectional attention over the image (the reference splits
         * by cu_seqlens; one image per call = the whole sequence). Per-head
         * contiguous copies let the score/prob GEMMs go through BLAS. */
        float scale = 1.0f / sqrtf((float)hd);
        for (int hq = 0; hq < m->heads; hq++) {
            for (int p = 0; p < n; p++) {
                const float *r = m->qkv + (size_t)p * 3 * hid + (size_t)hq * hd;
                memcpy(m->qc + (size_t)p * hd, r, hd * sizeof(float));
                memcpy(m->kc + (size_t)p * hd, r + hid, hd * sizeof(float));
                memcpy(m->vc + (size_t)p * hd, r + 2 * hid, hd * sizeof(float));
            }
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, hd,
                        scale, m->qc, hd, m->kc, hd, 0.0f, m->sc, n);
#else
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    const float *qi = m->qc + (size_t)i * hd;
                    const float *kj = m->kc + (size_t)j * hd;
                    float acc = 0.0f;
                    for (int d = 0; d < hd; d++) acc += qi[d] * kj[d];
                    m->sc[(size_t)i * n + j] = acc * scale;
                }
#endif
            for (int i = 0; i < n; i++) {
                float *row = m->sc + (size_t)i * n;
                float mx = row[0], sum = 0.0f;
                for (int j = 1; j < n; j++) if (row[j] > mx) mx = row[j];
                for (int j = 0; j < n; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
                float inv = 1.0f / sum;
                for (int j = 0; j < n; j++) row[j] *= inv;
            }
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, hd, n,
                        1.0f, m->sc, n, m->vc, hd, 0.0f,
                        m->att + (size_t)hq * hd, hid);
#else
            for (int i = 0; i < n; i++) {
                float *o = m->att + (size_t)i * hid + (size_t)hq * hd;
                for (int d = 0; d < hd; d++) {
                    float acc = 0.0f;
                    for (int j = 0; j < n; j++) acc += m->sc[(size_t)i * n + j] * m->vc[(size_t)j * hd + d];
                    o[d] = acc;
                }
            }
#endif
        }
        qwv_gemm(m->att, b->pw, b->pb, m->fc1, n, hid, hid);
        for (size_t i = 0; i < nx; i++) m->x[i] += m->fc1[i];

        qwv_ln(m->x, m->nb, n, hid, b->ln2w, b->ln2b);
        qwv_gemm(m->nb, b->f1w, b->f1b, m->fc1, n, hid, m->mlp);
        qwv_gelu_tanh(m->fc1, (size_t)n * m->mlp);
        qwv_gemm(m->fc1, b->f2w, b->f2b, m->nb, n, m->mlp, hid);
        for (size_t i = 0; i < nx; i++) m->x[i] += m->nb[i];

        /* deepstack snapshot */
        if (deepstack_out) {
            for (int d = 0; d < m->n_ds; d++)
                if (m->ds_idx[d] == l)
                    qwv_merge(m, &m->ds_w[d], m->x, n, 1,
                              deepstack_out + (size_t)d * (n / m->mlp_m) * m->out_h);
        }
    }

    qwv_merge(m, &m->merger_w, m->x, n, 0, merged);
    return 0;
}
