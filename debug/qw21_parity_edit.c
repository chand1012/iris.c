/* Editing-path parity: feed the reference dump (emb + ref latents + noise +
 * sigmas + img_mask) through qw_transformer_forward_edit and compare the
 * step-0 velocity to step0_v.bin. Confirms the joint sequence layout,
 * block-causal attention, per-block RoPE and per-token modulation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct qw_transformer qw_transformer_t;
extern qw_transformer_t *qw_transformer_load(const char *dir, int dim, int heads,
                                             int head_dim, int n_layers, int in_ch,
                                             int ctx_dim, int mlp_ratio,
                                             const int axes[3], float eps, int causal);
extern void qw_transformer_forward_edit(qw_transformer_t *tf, const float *z,
                                        const int *shapes, int n_shapes,
                                        const float *txt, const int *txt_mask,
                                        const int *img_mask, int n_slots,
                                        float sigma, float *v, int cache_mode);

#ifdef IRIS_DEV
extern void iris_dev_init(void);
extern int iris_dev_available(void);
#endif

static float *read_f32(const char *p, size_t *n)
{
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    float *b = malloc(sz);
    if (fread(b, 1, sz, f) != (size_t)sz) { fprintf(stderr, "short read %s\n", p); exit(1); }
    fclose(f); *n = sz / sizeof(float); return b;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <tfdir> <dumpdir> [out]\n", argv[0]); return 1; }
    char p[1024];
    static const int axes[3] = { 16, 56, 56 };
#ifdef IRIS_DEV
    iris_dev_init();
    fprintf(stderr, "device available: %d\n", iris_dev_available());
#endif
    qw_transformer_t *tf = qw_transformer_load(argv[1], 4096, 32, 128, 32, 64,
                                               4096, 3, axes, 1e-6f, 1);
    if (!tf) return 1;

    size_t n;
    float *emb = read_f32((snprintf(p, sizeof p, "%s/emb.bin", argv[2]), p), &n);
    int ctx = 4096;
    int n_emb = (int)(n / ctx);
    float *maskf = read_f32((snprintf(p, sizeof p, "%s/img_mask_full.bin", argv[2]), p), &n);
    int n_slots = (int)n;
    int *img_mask = malloc(n_slots * sizeof(int));
    for (int i = 0; i < n_slots; i++) img_mask[i] = maskf[i] != 0.0f;
    int n_img_slots = 0;
    for (int i = 0; i < n_slots; i++) n_img_slots += img_mask[i];
    int n_txt = n_slots - n_img_slots;
    if (n_emb != n_slots - 64) { fprintf(stderr, "emb rows %d, expected %d\n", n_emb, n_slots - 64); return 1; }
    int *txt_mask = malloc(n_emb * sizeof(int));
    for (int i = 0; i < n_emb; i++) txt_mask[i] = 1;

    float *ref = read_f32((snprintf(p, sizeof p, "%s/ref0.bin", argv[2]), p), &n);
    int n_ref = (int)(n / 64);
    float *noise = read_f32((snprintf(p, sizeof p, "%s/noise.bin", argv[2]), p), &n);
    int n_target = (int)(n / 64);

    int shapes[6];   /* one reference + target */
    int gh = (int)(sqrtf((float)n_ref) + 0.5f);
    shapes[0] = 1; shapes[1] = gh; shapes[2] = gh;
    int th = (int)(sqrtf((float)n_target) + 0.5f);
    shapes[3] = 1; shapes[4] = th; shapes[5] = th;
    fprintf(stderr, "ref grid %dx%d, target %dx%d, slots %d (txt %d, img slots %d)\n",
            gh, gh, th, th, n_slots, n_txt, n_img_slots);

    float *z = malloc((size_t)(n_ref + n_target) * 64 * sizeof(float));
    memcpy(z, ref, (size_t)n_ref * 64 * sizeof(float));
    memcpy(z + (size_t)n_ref * 64, noise, (size_t)n_target * 64 * sizeof(float));

    FILE *sf = fopen((snprintf(p, sizeof p, "%s/sigmas.txt", argv[2]), p), "r");
    float sig[64]; int ns = 0;
    while (ns < 64 && fscanf(sf, "%f", &sig[ns]) == 1) ns++;
    fclose(sf);

    float *v = malloc((size_t)n_target * 64 * sizeof(float));
    qw_transformer_forward_edit(tf, z, shapes, 2, emb, txt_mask, img_mask, n_slots,
                                sig[0], v, 0);

    float *vref = read_f32((snprintf(p, sizeof p, "%s/step0_v.bin", argv[2]), p), &n);
    if (argc > 3) { FILE *o = fopen(argv[3], "wb"); fwrite(v, 4, (size_t)n_target * 64, o); fclose(o); }

    size_t cnt = (size_t)n_target * 64;
    double mx = 0, mean = 0, rref = 0, rv = 0, dot = 0, nr = 0, nv = 0;
    int over = 0;
    for (size_t i = 0; i < cnt; i++) {
        double d = fabs(vref[i] - v[i]);
        if (d > mx) mx = d;
        if (d > 1e-3) over++;
        mean += d; rref += (double)vref[i] * vref[i]; rv += (double)v[i] * v[i];
        dot += (double)vref[i] * v[i]; nr += (double)vref[i] * vref[i]; nv += (double)v[i] * v[i];
    }
    fprintf(stderr, "EDIT_V n=%zu max_abs=%.3e mean_abs=%.3e over1e-3=%d (%.3f%%) | rms ref=%.4f ours=%.4f ratio=%.3f cos=%.5f\n",
            cnt, mx, mean / cnt, over, 100.0 * over / cnt,
            sqrt(rref / cnt), sqrt(rv / cnt), sqrt(rv / cnt) / sqrt(rref / cnt),
            dot / (sqrt(nr) * sqrt(nv)));
    double cosv = dot / (sqrt(nr) * sqrt(nv));
    printf("edit step-0 velocity parity %s (max_abs %.3e, cos %.5f; limits 1e-3 / 0.999)\n",
           (cosv >= 0.999) ? "PASS" : "FAIL", mx, cosv);
    return cosv >= 0.999 ? 0 : 1;
}
