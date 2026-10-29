/*
 * Iris Qwen VAE - AutoencoderKLQwenImage21
 *
 * 4-channel RGBA in/out, 16x spatial compression, 64 latent channels.
 * Weights are plain F32 safetensors.
 *
 * Encoder: conv_in 4->96; 5 down blocks over channels
 *          96, 96->192->384->768->768 (first four downsample 2x): each is
 *          2 residual blocks + stride-2 conv + avg-shortcut from the block
 *          input (AvgDown3D: channel/temporal slot grouping, mean over the
 *          group; image input pads one zero frame when factor_t = 2);
 *          mid = resnet -> single-head attention -> resnet (768);
 *          RMS_norm -> SiLU -> conv_out 768->128 -> quant_conv 1x1.
 *          The 128 moments split: first 64 are the mean; the latent is
 *          (mean - latents_mean) / latents_std (64 config values).
 *
 * Decoder: latents*std + mean -> post_quant_conv 1x1 -> conv_in 64->1152;
 *          mid = resnet -> attention -> resnet (1152); 5 up blocks over
 *          channels 1152, 1152->1152->576->288->144 (first four upsample
 *          2x spatially): each is 3 residual blocks + 2x-nearest/3x3-conv
 *          + dup-shortcut from the block input (DupUp3D, first_chunk
 *          keeps the last of factor_t frames); RMS_norm -> SiLU ->
 *          conv_out 144->4 -> clamp [-1, 1].
 *
 * Tiling follows the reference tiled_encode/tiled_decode: 256/192 sample
 * (16/12 latent) tiles, linear-ramp blend over the overlap, edge tiles
 * keep their (smaller) size.
 */

#include "iris.h"
#include "iris_safetensors.h"
#ifdef USE_METAL
#include "iris_metal.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QWVAE_Z       64
#define QWVAE_NBLOCK  5
#define QWVAE_TILE    256
#define QWVAE_TSTRIDE 192
#define QWVAE_LTILE   16
#define QWVAE_LSTRIDE 12

typedef struct {
    float *w, *b;     /* w: [oc, ic, kh, kw]; b: [oc] */
    int ic, oc, kh, kw;
} qw_conv_t;

typedef struct {
    qw_conv_t c1, c2, sc;   /* sc unused when ic == oc */
    float *g1, *g2;
    int ic, oc;
} qw_resnet_t;

typedef struct {
    float *g;
    qw_conv_t qkv, proj;
    int c;
} qw_attn_t;

typedef struct {
    qw_resnet_t res[2];
    qw_conv_t down;          /* 3x3 stride 2; unused in block 4 */
    int has_down;
    int ic, oc, ft, fs;      /* AvgDown3D factors (image: T=1) */
} qw_edown_t;

typedef struct {
    qw_resnet_t res[3];
    qw_conv_t up;            /* 3x3 after 2x nearest; unused in block 4 */
    int has_up;
    int ic, oc, ft, repeats; /* DupUp3D factors; repeats = oc*factor/ic */
} qw_dup_t;

struct iris_vae_qwen {
    qw_conv_t enc_in, enc_out, dec_in, dec_out, quant, post_quant;
    qw_resnet_t enc_mid_a, enc_mid_b, dec_mid_a, dec_mid_b;
    qw_attn_t enc_attn, dec_attn;
    float *enc_norm_g, *dec_norm_g;
    qw_edown_t edown[QWVAE_NBLOCK];
    qw_dup_t dup[QWVAE_NBLOCK];
    float lmean[QWVAE_Z], lstd[QWVAE_Z];
};

/* ----------------------------------------------------------------
 * Weight loading
 * ---------------------------------------------------------------- */

static float *get_f32(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "qwen vae: missing tensor %s\n", name);
        exit(1);
    }
    return safetensors_get_f32(sf, t);
}

static void load_conv(safetensors_file_t *sf, qw_conv_t *c, const char *name,
                      int ic, int oc, int kh, int kw) {
    char buf[300];
    snprintf(buf, sizeof(buf), "%s.weight", name);
    c->w = get_f32(sf, buf);
    snprintf(buf, sizeof(buf), "%s.bias", name);
    c->b = get_f32(sf, buf);
    c->ic = ic; c->oc = oc; c->kh = kh; c->kw = kw;
}

static void load_resnet(safetensors_file_t *sf, qw_resnet_t *r, const char *name,
                        int ic, int oc) {
    char buf[300];
    snprintf(buf, sizeof(buf), "%s.conv1", name);
    load_conv(sf, &r->c1, buf, ic, oc, 3, 3);
    snprintf(buf, sizeof(buf), "%s.conv2", name);
    load_conv(sf, &r->c2, buf, oc, oc, 3, 3);
    snprintf(buf, sizeof(buf), "%s.conv_shortcut", name);
    if (ic != oc) load_conv(sf, &r->sc, buf, ic, oc, 1, 1);
    else r->sc.w = NULL;
    snprintf(buf, sizeof(buf), "%s.norm1.gamma", name);
    r->g1 = get_f32(sf, buf);
    snprintf(buf, sizeof(buf), "%s.norm2.gamma", name);
    r->g2 = get_f32(sf, buf);
    r->ic = ic; r->oc = oc;
}

static void free_conv(qw_conv_t *c) { free(c->w); free(c->b); }

static void free_resnet(qw_resnet_t *r) {
    free_conv(&r->c1); free_conv(&r->c2); free_conv(&r->sc);
    free(r->g1); free(r->g2);
}

/* Read n floats following the first "[ ... ]" after key in buf. */
static int parse_float_array(const char *buf, const char *key, float *out, int n) {
    const char *p = strstr(buf, key);
    if (!p) return 0;
    p = strchr(p, '[');
    if (!p) return 0;
    p++;
    int i = 0;
    while (i < n && *p && *p != ']') {
        if (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r') { p++; continue; }
        out[i++] = (float)atof(p);
        p = strchr(p, ',');
        if (!p) break;
    }
    return i;
}

iris_vae_qwen_t *iris_vae_qwen_load(const char *vae_dir) {
    iris_vae_qwen_t *v = calloc(1, sizeof(*v));
    if (!v) return NULL;

    char path[1200];
    snprintf(path, sizeof(path), "%s/diffusion_pytorch_model.safetensors", vae_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "qwen vae: cannot open %s\n", path);
        free(v);
        return NULL;
    }

    /* latents_mean / latents_std from config.json */
    snprintf(path, sizeof(path), "%s/config.json", vae_dir);
    FILE *f = fopen(path, "r");
    if (f) {
        char buf[16384];
        size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        buf[n] = '\0';
        fclose(f);
        if (parse_float_array(buf, "\"latents_mean\"", v->lmean, QWVAE_Z) != QWVAE_Z ||
            parse_float_array(buf, "\"latents_std\"", v->lstd, QWVAE_Z) != QWVAE_Z) {
            fprintf(stderr, "qwen vae: latents_mean/latents_std not found in %s\n", path);
            safetensors_close(sf);
            free(v);
            return NULL;
        }
    } else {
        fprintf(stderr, "qwen vae: cannot open %s\n", path);
        safetensors_close(sf);
        free(v);
        return NULL;
    }

    load_conv(sf, &v->enc_in, "encoder.conv_in", 4, 96, 3, 3);
    load_conv(sf, &v->enc_out, "encoder.conv_out", 768, 128, 3, 3);
    load_conv(sf, &v->dec_in, "decoder.conv_in", 64, 1152, 3, 3);
    load_conv(sf, &v->dec_out, "decoder.conv_out", 144, 4, 3, 3);
    load_conv(sf, &v->quant, "quant_conv", 128, 128, 1, 1);
    load_conv(sf, &v->post_quant, "post_quant_conv", 64, 64, 1, 1);

    const int enc_ch[QWVAE_NBLOCK + 1] = {96, 96, 192, 384, 768, 768};
    const int dec_ch[QWVAE_NBLOCK + 1] = {1152, 1152, 1152, 576, 288, 144};
    char name[300];

    for (int i = 0; i < QWVAE_NBLOCK; i++) {
        qw_edown_t *d = &v->edown[i];
        d->ic = enc_ch[i]; d->oc = enc_ch[i + 1];
        d->ft = (i >= 1 && i <= 3) ? 2 : 1;   /* tds = [F,T,T,T]; block 4 has none */
        d->fs = i < 4 ? 2 : 1;
        for (int m = 0; m < 2; m++) {
            snprintf(name, sizeof(name), "encoder.down_blocks.%d.resnets.%d", i, m);
            load_resnet(sf, &d->res[m], name, enc_ch[i + m], enc_ch[i + 1]);
        }
        d->has_down = (i < 4);
        if (i < 4) {
            snprintf(name, sizeof(name), "encoder.down_blocks.%d.downsampler.resample.1", i);
            load_conv(sf, &d->down, name, enc_ch[i + 1], enc_ch[i + 1], 3, 3);
        }
    }

    for (int i = 0; i < QWVAE_NBLOCK; i++) {
        qw_dup_t *d = &v->dup[i];
        d->ic = dec_ch[i]; d->oc = dec_ch[i + 1];
        d->ft = i < 3 ? 2 : 1;               /* tus = [T,T,T,F] */
        d->repeats = i < 4 ? (i < 3 ? d->oc * 8 / d->ic : d->oc * 4 / d->ic) : 0;
        for (int m = 0; m < 3; m++) {
            snprintf(name, sizeof(name), "decoder.up_blocks.%d.resnets.%d", i, m);
            load_resnet(sf, &d->res[m], name,
                        m == 0 ? dec_ch[i] : dec_ch[i + 1], dec_ch[i + 1]);
        }
        d->has_up = (i < 4);
        if (i < 4) {
            snprintf(name, sizeof(name), "decoder.up_blocks.%d.upsampler.resample.1", i);
            load_conv(sf, &d->up, name, dec_ch[i + 1], dec_ch[i + 1], 3, 3);
        }
    }

    load_resnet(sf, &v->enc_mid_a, "encoder.mid_block.resnets.0", 768, 768);
    load_resnet(sf, &v->enc_mid_b, "encoder.mid_block.resnets.1", 768, 768);
    load_resnet(sf, &v->dec_mid_a, "decoder.mid_block.resnets.0", 1152, 1152);
    load_resnet(sf, &v->dec_mid_b, "decoder.mid_block.resnets.1", 1152, 1152);

    v->enc_attn.c = 768;
    v->enc_attn.g = get_f32(sf, "encoder.mid_block.attentions.0.norm.gamma");
    load_conv(sf, &v->enc_attn.qkv, "encoder.mid_block.attentions.0.to_qkv", 768, 2304, 1, 1);
    load_conv(sf, &v->enc_attn.proj, "encoder.mid_block.attentions.0.proj", 768, 768, 1, 1);

    v->dec_attn.c = 1152;
    v->dec_attn.g = get_f32(sf, "decoder.mid_block.attentions.0.norm.gamma");
    load_conv(sf, &v->dec_attn.qkv, "decoder.mid_block.attentions.0.to_qkv", 1152, 3456, 1, 1);
    load_conv(sf, &v->dec_attn.proj, "decoder.mid_block.attentions.0.proj", 1152, 1152, 1, 1);

    v->enc_norm_g = get_f32(sf, "encoder.norm_out.gamma");
    v->dec_norm_g = get_f32(sf, "decoder.norm_out.gamma");

    safetensors_close(sf);
    return v;
}

void iris_vae_qwen_free(iris_vae_qwen_t *v) {
    if (!v) return;
    free_conv(&v->enc_in); free_conv(&v->enc_out);
    free_conv(&v->dec_in); free_conv(&v->dec_out);
    free_conv(&v->quant); free_conv(&v->post_quant);
    free_resnet(&v->enc_mid_a); free_resnet(&v->enc_mid_b);
    free_resnet(&v->dec_mid_a); free_resnet(&v->dec_mid_b);
    free(v->enc_attn.g); free_conv(&v->enc_attn.qkv); free_conv(&v->enc_attn.proj);
    free(v->dec_attn.g); free_conv(&v->dec_attn.qkv); free_conv(&v->dec_attn.proj);
    free(v->enc_norm_g); free(v->dec_norm_g);
    for (int i = 0; i < QWVAE_NBLOCK; i++) {
        free_resnet(&v->edown[i].res[0]); free_resnet(&v->edown[i].res[1]);
        free_conv(&v->edown[i].down);
        for (int m = 0; m < 3; m++) free_resnet(&v->dup[i].res[m]);
        free_conv(&v->dup[i].up);
    }
    free(v);
}

static void qw_dump(const char *tag, const float *x, int c, int h, int w) {
    const char *dir = getenv("QW_VAE_DUMP");
    if (!dir) return;
    char path[600], hdr[128];
    int hl = snprintf(hdr, sizeof(hdr),
                      "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d, %d), }",
                      c, h, w);
    snprintf(path, sizeof(path), "%s/%s.npy", dir, tag);
    FILE *f = fopen(path, "wb");
    if (!f) return;
    const unsigned char magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    fwrite(magic, 1, 6, f);
    fputc(1, f); fputc(0, f);
    fputc(hl & 0xff, f); fputc((hl >> 8) & 0xff, f);
    fwrite(hdr, 1, hl, f);
    fwrite(x, sizeof(float), (size_t)c * h * w, f);
    fclose(f);
}

/* ----------------------------------------------------------------
 * Basic ops (NCHW layout, batch 1)
 * ---------------------------------------------------------------- */

/* 3x3 conv, padding 1 on all sides. */
static void qw_conv3(float *out, const float *in, int H, int W, qw_conv_t *c) {
    int ic = c->ic, oc = c->oc;
#ifdef USE_METAL
    if (!iris_metal_available()) {
        iris_metal_init();
    }
    if (iris_metal_available() &&
        iris_metal_conv2d(out, in, c->w, c->b, 1, ic, oc, H, W, 3, 3, 1, 1)) {
        return;
    }
#endif
    float *p = malloc((size_t)(H + 2) * (W + 2) * ic * sizeof(float));
    for (int ch = 0; ch < ic; ch++) {
        float *dst = p + (size_t)ch * (H + 2) * (W + 2);
        memset(dst, 0, (size_t)(H + 2) * (W + 2) * sizeof(float));
        for (int y = 0; y < H; y++)
            memcpy(dst + (size_t)(y + 1) * (W + 2) + 1,
                   in + (size_t)(ch * H + y) * W, (size_t)W * sizeof(float));
    }
    for (int o = 0; o < oc; o++) {
        float *od = out + (size_t)o * H * W;
        const float *w = c->w + (size_t)o * ic * 9;
        for (int j = 0; j < H * W; j++) od[j] = c->b[o];
        for (int i = 0; i < ic; i++) {
            const float *pi = p + (size_t)i * (H + 2) * (W + 2);
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    float wv = w[i * 9 + ky * 3 + kx];
                    const float *pr = pi + (size_t)ky * (W + 2) + kx;
                    for (int y = 0; y < H; y++) {
                        float *odrow = od + (size_t)y * W;
                        const float *prow = pr + (size_t)y * (W + 2);
                        for (int x = 0; x < W; x++)
                            odrow[x] += wv * prow[x];
                    }
                }
        }
    }
    free(p);
}

/* 3x3 stride-2 conv with zero padding (0, 0, 1, 1): right and bottom only.
 * A window row/col at the H/W boundary of the unpadded input reads zero. */
static void qw_conv3s2(float *out, const float *in, int H, int W, qw_conv_t *c) {
    int ic = c->ic, oc = c->oc;
    int Ho = H / 2, Wo = W / 2;
    for (int o = 0; o < oc; o++) {
        float *od = out + (size_t)o * Ho * Wo;
        const float *w = c->w + (size_t)o * ic * 9;
        for (int j = 0; j < Ho * Wo; j++) od[j] = c->b[o];
        for (int i = 0; i < ic; i++) {
            const float *pi = in + (size_t)i * H * W;
            for (int ky = 0; ky < 3; ky++) {
                int ymax = (H - 1 - ky) / 2 + 1; /* rows with 2y+ky < H */
                if (ymax > Ho) ymax = Ho;
                for (int kx = 0; kx < 3; kx++) {
                    float wv = w[i * 9 + ky * 3 + kx];
                    int xmax = (W - 1 - kx) / 2 + 1;
                    if (xmax > Wo) xmax = Wo;
                    for (int y = 0; y < ymax; y++) {
                        float *odrow = od + (size_t)y * Wo;
                        const float *prow = pi + (size_t)(2 * y + ky) * W + kx;
                        for (int x = 0; x < xmax; x++)
                            odrow[x] += wv * prow[2 * x];
                    }
                }
            }
        }
    }
}

/* 1x1 conv over the channel axis (in/out laid out [c, n]). */
static void qw_conv1(float *out, const float *in, int n, qw_conv_t *c) {
    int ic = c->ic, oc = c->oc;
#ifdef USE_METAL
    if (!iris_metal_available()) {
        iris_metal_init();
    }
    if (iris_metal_available() &&
        iris_metal_conv2d(out, in, c->w, c->b, 1, ic, oc, n, 1, 1, 1, 1, 0)) {
        return;
    }
#endif
    for (int o = 0; o < oc; o++) {
        float *od = out + (size_t)o * n;
        const float *w = c->w + (size_t)o * ic;
        for (int j = 0; j < n; j++) od[j] = c->b[o];
        for (int i = 0; i < ic; i++) {
            float wv = w[i];
            const float *ir = in + (size_t)i * n;
            for (int j = 0; j < n; j++)
                od[j] += wv * ir[j];
        }
    }
}

/* RMS normalization over channels, per pixel: x *= sqrt(c)/rms * gamma
 * (F.normalize dim=1, eps 1e-12, then scale sqrt(c) and gamma). */
static void qw_rms(float *x, int H, int W, int c, const float *g) {
    float scale = sqrtf((float)c);
    for (int j = 0; j < H * W; j++) {
        float s = 0;
        float *row = x + j;
        for (int i = 0; i < c; i++) {
            float t = row[(size_t)i * H * W];
            s += t * t;
        }
        float d = sqrtf(s);
        if (d < 1e-12f) d = 1e-12f;
        float k = scale / d;
        for (int i = 0; i < c; i++) row[(size_t)i * H * W] *= k * g[i];
    }
}

static void qw_silu(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}

/* Single-head full HxW self-attention with 1x1 qkv/proj convs, residual. */
static void qw_attn(iris_vae_qwen_t *v, float *x, int H, int W, qw_attn_t *a) {
    int c = a->c, hw = H * W;
    float *xm = malloc((size_t)c * hw * sizeof(float));
    float *qkv = malloc((size_t)3 * c * hw * sizeof(float));
    float *s = malloc((size_t)hw * hw * sizeof(float));
    float *o = malloc((size_t)c * hw * sizeof(float));
    memcpy(xm, x, (size_t)c * hw * sizeof(float));
    qw_rms(xm, H, W, c, a->g);
    qw_conv1(qkv, xm, hw, &a->qkv);
    free(xm);

    const float *q = qkv, *k = qkv + (size_t)c * hw, *vv = qkv + (size_t)2 * c * hw;
    float inv = 1.0f / sqrtf((float)c);
    for (int i = 0; i < hw; i++) {
        float m = -1e30f;
        for (int j = 0; j < hw; j++) {
            float dot = 0;
            for (int d = 0; d < c; d++)
                dot += q[(size_t)d * hw + i] * k[(size_t)d * hw + j];
            float t = dot * inv;
            s[(size_t)i * hw + j] = t;
            if (t > m) m = t;
        }
        float sum = 0;
        for (int j = 0; j < hw; j++) {
            s[(size_t)i * hw + j] = expf(s[(size_t)i * hw + j] - m);
            sum += s[(size_t)i * hw + j];
        }
        for (int j = 0; j < hw; j++) s[(size_t)i * hw + j] /= sum;
    }
    for (int i = 0; i < hw; i++)
        for (int d = 0; d < c; d++) {
            float acc = 0;
            const float *sr = s + (size_t)i * hw;
            const float *vr = vv + (size_t)d * hw;
            for (int j = 0; j < hw; j++) acc += sr[j] * vr[j];
            o[(size_t)d * hw + i] = acc;
        }
    free(qkv);
    free(s);
    float *po = malloc((size_t)c * hw * sizeof(float));
    qw_conv1(po, o, hw, &a->proj); /* not in-place: conv1 reads/writes interleaved */
    free(o);
    for (size_t i = 0; i < (size_t)c * hw; i++) x[i] += po[i];
    free(po);
}

/* Residual block: h = sc(x) [or x]; x = conv2(silu(rms(g2)(conv1(
 * silu(rms(g1)(x)))))); out = x + h. norm1 uses the input channels. */
static void qw_resnet(iris_vae_qwen_t *v, float *out, const float *in, int H, int W,
                      qw_resnet_t *r) {
    int ic = r->ic, oc = r->oc;
    float *h = malloc((size_t)oc * H * W * sizeof(float));
    float *t = malloc((size_t)ic * H * W * sizeof(float));
    if (r->sc.w) qw_conv1(h, in, H * W, &r->sc);
    else memcpy(h, in, (size_t)ic * H * W * sizeof(float));
    memcpy(t, in, (size_t)ic * H * W * sizeof(float));
    qw_rms(t, H, W, ic, r->g1);
    qw_silu(t, (size_t)ic * H * W);
    float *t2 = malloc((size_t)oc * H * W * sizeof(float));
    qw_conv3(t2, t, H, W, &r->c1);
    free(t);
    qw_rms(t2, H, W, oc, r->g2);
    qw_silu(t2, (size_t)oc * H * W);
    qw_conv3(t2, t2, H, W, &r->c2);
    for (size_t i = 0; i < (size_t)oc * H * W; i++) t2[i] += h[i];
    memcpy(out, t2, (size_t)oc * H * W * sizeof(float));
    free(h);
    free(t2);
}

/* AvgDown3D for image input (T=1, padded to ft frames; only frame 0 is
 * real). Slot s = c*(ft*fs*fs) + t'*fs*fs + p1*fs + p2; out channel o
 * averages the group = ic*factor/oc slots [o*group, (o+1)*group) -- for
 * images the padded (t' > 0) slots contribute zero. */
static void qw_avgdown(float *out, const float *in, int H, int W,
                       int ic, int oc, int ft, int fs) {
    int factor = ft * fs * fs;
    int g = ic * factor / oc;
    int Ho = H / fs, Wo = W / fs;
    for (int o = 0; o < oc; o++)
        for (int y = 0; y < Ho; y++)
            for (int x = 0; x < Wo; x++) {
                float acc = 0;
                for (int j = 0; j < g; j++) {
                    int s = o * g + j;
                    int c = s / factor, rem = s % factor;
                    int t2 = rem / (fs * fs);
                    int p1 = (rem / fs) % fs, p2 = rem % fs;
                    /* F.pad pads T on the front: the real frame is ft-1, the pad is 0 */
                    if (t2 == ft - 1)
                        acc += in[(size_t)(c * H + y * fs + p1) * W + x * fs + p2];
                }
                out[(size_t)o * Ho * Wo + y * Wo + x] = acc / (float)g;
            }
}

/* DupUp3D for images (first_chunk keeps the last of the ft frames):
 * out[o][2y+p1][2x+p2] = in[c][y][x] with
 * c = (o*factor + (ft-1)*4 + p1*2 + p2) / repeats, factor = ft*4. */
static void qw_dupup(float *out, const float *in, int H, int W,
                     int oc, int ft, int repeats) {
    int factor = ft * 4, Ho = 2 * H, Wo = 2 * W;
    for (int o = 0; o < oc; o++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                for (int p1 = 0; p1 < 2; p1++)
                    for (int p2 = 0; p2 < 2; p2++) {
                        int c = (o * factor + (ft - 1) * 4 + p1 * 2 + p2) / repeats;
                        out[(size_t)(o * Ho + y * 2 + p1) * Wo + x * 2 + p2] =
                            in[(size_t)(c * H + y) * W + x];
                    }
            }
}

/* Nearest 2x upsample (nearest-exact at integer scale). */
static void qw_up2x(float *out, const float *in, int H, int W, int c) {
    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                float val = in[(size_t)(ch * H + y) * W + x];
                out[(size_t)(ch * 2 * H + y * 2) * W * 2 + x * 2] = val;
                out[(size_t)(ch * 2 * H + y * 2) * W * 2 + x * 2 + 1] = val;
                out[(size_t)(ch * 2 * H + y * 2 + 1) * W * 2 + x * 2] = val;
                out[(size_t)(ch * 2 * H + y * 2 + 1) * W * 2 + x * 2 + 1] = val;
            }
}

static void qw_clamp(float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = x[i] < -1.0f ? -1.0f : (x[i] > 1.0f ? 1.0f : x[i]);
}

/* ----------------------------------------------------------------
 * Encoder / decoder (un-tiled core)
 * ---------------------------------------------------------------- */

/* rgba [4,H,W] in [-1,1] -> moments [128,H/16,W/16]. */
static float *qw_encode_core(iris_vae_qwen_t *v, const float *img, int H, int W) {
    float *x = malloc((size_t)96 * H * W * sizeof(float));
    qw_conv3(x, img, H, W, &v->enc_in);
    qw_dump("c_enc_conv_in", x, 96, H, W);

    for (int i = 0; i < QWVAE_NBLOCK; i++) {
        qw_edown_t *d = &v->edown[i];
        int Hin = H, Win = W;
        float *xc = malloc((size_t)d->ic * Hin * Win * sizeof(float));
        memcpy(xc, x, (size_t)d->ic * Hin * Win * sizeof(float));
        float *x2 = malloc((size_t)d->oc * H * W * sizeof(float));
        qw_resnet(v, x2, x, H, W, &d->res[0]);
        free(x);
        x = x2;
        qw_resnet(v, x, x, H, W, &d->res[1]);
        if (d->has_down) {
            float *t = malloc((size_t)d->oc * H / 2 * W / 2 * sizeof(float));
            qw_conv3s2(t, x, H, W, &d->down);
            free(x);
            x = t;
            H /= 2; W /= 2;
        }
        float *s = malloc((size_t)d->oc * H * W * sizeof(float));
        qw_avgdown(s, xc, Hin, Win, d->ic, d->oc, d->ft, d->fs);
        for (size_t k = 0; k < (size_t)d->oc * H * W; k++) x[k] += s[k];
        free(s);
        free(xc);
        char tag[32];
        snprintf(tag, sizeof(tag), "c_enc_b%d", i);
        qw_dump(tag, x, d->oc, H, W);
    }

    qw_resnet(v, x, x, H, W, &v->enc_mid_a);
    qw_attn(v, x, H, W, &v->enc_attn);
    qw_resnet(v, x, x, H, W, &v->enc_mid_b);
    qw_dump("c_enc_mid", x, 768, H, W);
    qw_rms(x, H, W, 768, v->enc_norm_g);
    qw_silu(x, (size_t)768 * H * W);
    float *m = malloc((size_t)128 * H * W * sizeof(float));
    qw_conv3(m, x, H, W, &v->enc_out);
    qw_dump("c_enc_convout", m, 128, H, W);
    free(x);
    float *mom = malloc((size_t)128 * H * W * sizeof(float));
    qw_conv1(mom, m, H * W, &v->quant);
    free(m);
    return mom;
}

/* Normalized latents [64,lh,lw] -> rgba [4,16lh,16lw] clamped to [-1,1]. */
static float *qw_decode_core(iris_vae_qwen_t *v, const float *lat, int lh, int lw) {
    int H = lh, W = lw;
    float *x = malloc((size_t)QWVAE_Z * H * W * sizeof(float));
    for (int ch = 0; ch < QWVAE_Z; ch++)
        for (int i = 0; i < H * W; i++)
            x[(size_t)ch * H * W + i] = lat[(size_t)ch * H * W + i] * v->lstd[ch] + v->lmean[ch];
    float *t = malloc((size_t)QWVAE_Z * H * W * sizeof(float));
    qw_conv1(t, x, H * W, &v->post_quant);
    free(x);
    x = t;

    float *x2 = malloc((size_t)1152 * H * W * sizeof(float));
    qw_conv3(x2, x, H, W, &v->dec_in);
    qw_dump("c_dec_convin", x2, 1152, H, W);
    free(x);
    x = x2;

    qw_resnet(v, x, x, H, W, &v->dec_mid_a);
    qw_attn(v, x, H, W, &v->dec_attn);
    qw_resnet(v, x, x, H, W, &v->dec_mid_b);
    qw_dump("c_dec_mid", x, 1152, H, W);

    for (int i = 0; i < QWVAE_NBLOCK; i++) {
        qw_dup_t *d = &v->dup[i];
        float *xc = malloc((size_t)d->ic * H * W * sizeof(float));
        memcpy(xc, x, (size_t)d->ic * H * W * sizeof(float));
        float *x2 = malloc((size_t)d->oc * H * W * sizeof(float));
        qw_resnet(v, x2, x, H, W, &d->res[0]);
        free(x);
        x = x2;
        qw_resnet(v, x, x, H, W, &d->res[1]);
        qw_resnet(v, x, x, H, W, &d->res[2]);
        if (d->has_up) {
            int Ho = H * 2, Wo = W * 2;
            float *up = malloc((size_t)d->oc * Ho * Wo * sizeof(float));
            qw_up2x(up, x, H, W, d->oc);
            qw_conv3(up, up, Ho, Wo, &d->up);
            float *s = malloc((size_t)d->oc * Ho * Wo * sizeof(float));
            qw_dupup(s, xc, H, W, d->oc, d->ft, d->repeats);
            for (size_t k = 0; k < (size_t)d->oc * Ho * Wo; k++) up[k] += s[k];
            free(s);
            free(x);
            x = up;
            H = Ho; W = Wo;
        }
        free(xc);
        char tag[32];
        snprintf(tag, sizeof(tag), "c_dec_u%d", i);
        qw_dump(tag, x, d->oc, H, W);
    }

    qw_rms(x, H, W, 144, v->dec_norm_g);
    qw_silu(x, (size_t)144 * H * W);
    float *img = malloc((size_t)4 * H * W * sizeof(float));
    qw_conv3(img, x, H, W, &v->dec_out);
    free(x);
    qw_dump("c_dec_convout", img, 4, H, W);
    qw_clamp(img, (size_t)4 * H * W);
    return img;
}

/* ----------------------------------------------------------------
 * Tiled paths
 * ---------------------------------------------------------------- */

/* Linear-ramp vertical blend: b's first e rows toward a's bottom rows. */
static void qw_blend_v(float *a, int ah, int w, float *b, int bh, int c, int e) {
    e = e < ah ? e : ah;
    e = e < bh ? e : bh;
    for (int y = 0; y < e; y++) {
        float wa = 1.0f - y / (float)e;
        float wb = y / (float)e;
        for (int ch = 0; ch < c; ch++) {
            const float *ra = a + ((size_t)ch * ah + ah - e + y) * w;
            float *rb = b + ((size_t)ch * bh + y) * w;
            for (int x = 0; x < w; x++) rb[x] = ra[x] * wa + rb[x] * wb;
        }
    }
}

/* Linear-ramp horizontal blend: b's first e cols toward a's right cols. */
static void qw_blend_h(float *a, int ah, int aw, float *b, int bw, int c, int e) {
    e = e < aw ? e : aw;
    e = e < bw ? e : bw;
    for (int x = 0; x < e; x++) {
        float wa = 1.0f - x / (float)e;
        float wb = x / (float)e;
        for (int ch = 0; ch < c; ch++)
            for (int y = 0; y < ah; y++) {
                const float *ra = a + ((size_t)ch * ah + y) * aw;
                float *rb = b + ((size_t)ch * ah + y) * bw;
                for (int xx = 0; xx < e; xx++)
                    rb[xx] = ra[aw - e + xx] * wa + rb[xx] * wb;
            }
    }
}

float *iris_vae_qwen_encode(iris_vae_qwen_t *v, const float *rgba, int H, int W,
                            int tile, int *out_h, int *out_w) {
    int lh = H / 16, lw = W / 16;
    *out_h = lh;
    *out_w = lw;

    float *mom;
    if (!tile || (H <= QWVAE_TILE && W <= QWVAE_TILE)) {
        mom = qw_encode_core(v, rgba, H, W);
    } else {
        /* latent-space tiling: 16/12 tiles, blend 4 */
        mom = calloc(1, (size_t)128 * lh * lw * sizeof(float));
        int ncol = (W + QWVAE_TSTRIDE - 1) / QWVAE_TSTRIDE;
        float **prev = calloc(ncol, sizeof(*prev));
        int *prev_lh = calloc(ncol, sizeof(int));
        for (int top = 0; top < H; top += QWVAE_TSTRIDE) {
            int th = QWVAE_TILE < H - top ? QWVAE_TILE : H - top;
            int tlh = th / 16;
            float **row = calloc(ncol, sizeof(*row));
            int *row_lh = calloc(ncol, sizeof(int));
            for (int left = 0; left < W; left += QWVAE_TSTRIDE) {
                int idx = left / QWVAE_TSTRIDE;
                int tw = QWVAE_TILE < W - left ? QWVAE_TILE : W - left;
                int tlw = tw / 16;
                float *timg = malloc((size_t)4 * th * tw * sizeof(float));
                for (int ch = 0; ch < 4; ch++)
                    for (int y = 0; y < th; y++)
                        memcpy(timg + ((size_t)ch * th + y) * tw,
                               rgba + ((size_t)ch * H + top + y) * W + left,
                               (size_t)tw * sizeof(float));
                float *tl = qw_encode_core(v, timg, th, tw);
                free(timg);
                if (top > 0 && prev[idx])
                    qw_blend_v(prev[idx], prev_lh[idx], tlw, tl, tlh, 128,
                               QWVAE_LTILE - QWVAE_LSTRIDE);
                if (left > 0 && row[idx - 1])
                    qw_blend_h(row[idx - 1], row_lh[idx - 1], tlw, tl, tlw, 128,
                               QWVAE_LTILE - QWVAE_LSTRIDE);
                /* keep the leading stride of the (already blended) tile */
                int keep = QWVAE_LSTRIDE < tlh ? QWVAE_LSTRIDE : tlh;
                int keepw = QWVAE_LSTRIDE < tlw ? QWVAE_LSTRIDE : tlw;
                for (int ch = 0; ch < 128; ch++)
                    for (int y = 0; y < keep; y++)
                        memcpy(mom + ((size_t)ch * lh + top / 16 + y) * lw + left / 16,
                               tl + ((size_t)ch * tlh + y) * tlw,
                               (size_t)keepw * sizeof(float));
                row[idx] = tl;
                row_lh[idx] = tlh;
            }
            for (int idx = 0; idx < ncol; idx++) free(row[idx]);
            free(row);
            free(prev);
            prev = row;   /* current row becomes the previous one */
            free(prev_lh);
            prev_lh = row_lh;
        }
        free(prev);
        free(prev_lh);
    }

    /* split moments -> mean, normalize */
    float *lat = malloc((size_t)QWVAE_Z * lh * lw * sizeof(float));
    for (int ch = 0; ch < QWVAE_Z; ch++)
        for (int i = 0; i < lh * lw; i++)
            lat[(size_t)ch * lh * lw + i] =
                (mom[(size_t)ch * lh * lw + i] - v->lmean[ch]) / v->lstd[ch];
    free(mom);
    return lat;
}

float *iris_vae_qwen_decode(iris_vae_qwen_t *v, const float *latents, int lh, int lw,
                            int tile, int *out_h, int *out_w) {
    int H = lh * 16, W = lw * 16;
    *out_h = H;
    *out_w = W;

    if (!tile || (lh <= QWVAE_LTILE && lw <= QWVAE_LTILE))
        return qw_decode_core(v, latents, lh, lw);

    /* sample-space tiling: 256/192 tiles, blend 64 */
    float *img = calloc(1, (size_t)4 * H * W * sizeof(float));
    int ncol = (lw + QWVAE_LSTRIDE - 1) / QWVAE_LSTRIDE;
    float **prev = calloc(ncol, sizeof(*prev));
    int *prev_sh = calloc(ncol, sizeof(int));
    for (int top = 0; top < lh; top += QWVAE_LSTRIDE) {
        int th = QWVAE_LTILE < lh - top ? QWVAE_LTILE : lh - top;
        int tsh = th * 16;
        float **row = calloc(ncol, sizeof(*row));
        int *row_sh = calloc(ncol, sizeof(int));
        for (int left = 0; left < lw; left += QWVAE_LSTRIDE) {
            int idx = left / QWVAE_LSTRIDE;
            int tw = QWVAE_LTILE < lw - left ? QWVAE_LTILE : lw - left;
            int tsw = tw * 16;
            float *tl = malloc((size_t)QWVAE_Z * th * tw * sizeof(float));
            for (int ch = 0; ch < QWVAE_Z; ch++)
                for (int y = 0; y < th; y++)
                    memcpy(tl + ((size_t)ch * th + y) * tw,
                           latents + ((size_t)ch * lh + top + y) * lw + left,
                           (size_t)tw * sizeof(float));
            float *ti = qw_decode_core(v, tl, th, tw);
            free(tl);
            if (top > 0 && prev[idx])
                qw_blend_v(prev[idx], prev_sh[idx], tsw, ti, tsh, 4,
                           QWVAE_TILE - QWVAE_TSTRIDE);
            if (left > 0 && row[idx - 1])
                qw_blend_h(row[idx - 1], row_sh[idx - 1], tsw, ti, tsw, 4,
                           QWVAE_TILE - QWVAE_TSTRIDE);
            int keep = QWVAE_TSTRIDE < tsh ? QWVAE_TSTRIDE : tsh;
            int keepw = QWVAE_TSTRIDE < tsw ? QWVAE_TSTRIDE : tsw;
            for (int ch = 0; ch < 4; ch++)
                for (int y = 0; y < keep; y++)
                    memcpy(img + ((size_t)ch * H + top * 16 + y) * W + left * 16,
                           ti + ((size_t)ch * tsh + y) * tsw,
                           (size_t)keepw * sizeof(float));
            row[idx] = ti;
            row_sh[idx] = tsh;
        }
        for (int idx = 0; idx < ncol; idx++) free(row[idx]);
        free(row);
        free(prev);
        prev = row;
        free(prev_sh);
        prev_sh = row_sh;
    }
    free(prev);
    free(prev_sh);
    qw_clamp(img, (size_t)4 * H * W);
    return img;
}
