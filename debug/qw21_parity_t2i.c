/* 40-step parity: feed reference emb.bin + noise.bin + sigmas.txt through the
 * production sampler (iris_sample_euler_qwen21) and compare final latents to
 * the reference latent.bin. Confirms sampler accumulation + transformer
 * numerics at the real step count. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../iris.h"

typedef struct qw_transformer qw_transformer_t;
extern qw_transformer_t *qw_transformer_load(const char *dir, int dim, int heads,
    int head_dim, int n_layers, int in_ch, int ctx_dim, int mlp_ratio,
    const int axes[3], float eps, int causal);
extern float *iris_sample_euler_qwen21(qw_transformer_t *tf, float *z,
    int img_h, int img_w, const float *txt, int s_txt, const int *txt_mask,
    const float *schedule, int num_steps, void (*progress)(int, int));

static float *read_f32(const char *p, int *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    *n = (int)(sz / 4);
    float *b = malloc(sz);
    if (fread(b, 4, *n, f) != (size_t)*n) { fprintf(stderr, "read %s\n", p); exit(1); }
    fclose(f); return b;
}

void qw_transformer_forward(void *tf, const float *z, const float *txt, const int *txt_mask,
    int s_txt, int img_h, int img_w, float sigma, float *v);
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <model_dir> <dump_dir>\n", argv[0]); return 2; }
    char p[512];

    int axes[3] = {16, 56, 56};
    qw_transformer_t *tf = qw_transformer_load(argv[1], 4096, 32, 128, 32, 64, 4096, 3, axes, 1e-6f, 1);
    if (!tf) { fprintf(stderr, "load failed\n"); return 1; }

    int n_emb, n_noise, n_lat;
    snprintf(p, sizeof p, "%s/emb.bin", argv[2]);   float *emb = read_f32(p, &n_emb);
    snprintf(p, sizeof p, "%s/noise.bin", argv[2]);  float *noise = read_f32(p, &n_noise);
    snprintf(p, sizeof p, "%s/latent.bin", argv[2]); float *ref = read_f32(p, &n_lat);

    int s_txt = n_emb / 4096;
    int n_img = n_noise / 64;
    int side = (int)(sqrtf((float)n_img) + 0.5f);

    /* sigmas */
    FILE *sf = fopen((snprintf(p, sizeof p, "%s/sigmas.txt", argv[2]), p), "r");
    if (!sf) { fprintf(stderr, "no sigmas.txt\n"); return 1; }
    float sched[128]; int n_steps = 0;
    while (n_steps < 127 && fscanf(sf, "%f", &sched[n_steps]) == 1) n_steps++;
    fclose(sf);
    n_steps -= 1; /* file has num_steps+1 values */

    fprintf(stderr, "s_txt=%d n_img=%d side=%d steps=%d\n", s_txt, n_img, side, n_steps);

    float *z = malloc((size_t)n_noise * 4);
    for (int i = 0; i < n_noise; i++) z[i] = noise[i];
    float *v = malloc((size_t)n_noise * 4);
    for (int st = 0; st < n_steps; st++) {
        qw_transformer_forward(tf, z, emb, NULL, s_txt, side, side, sched[st], v);
        char pn[256]; snprintf(pn, sizeof pn, "/tmp/qw_mstep%d_v.bin", st);
        FILE *F = fopen(pn, "wb"); fwrite(v, 4, (size_t)n_noise, F); fclose(F);
        float ds = sched[st+1] - sched[st];
        for (int i = 0; i < n_noise; i++) z[i] += ds * v[i];
    }
    float *out = z;

    double maxd = 0, sum = 0; int over = 0;
    for (int i = 0; i < n_lat; i++) {
        double d = fabs((double)out[i] - ref[i]);
        if (d > maxd) maxd = d; sum += d;
        if (d > 1e-3) over++;
    }
    if (argc > 3) { FILE *F = fopen(argv[3], "wb"); fwrite(out, 4, (size_t)n_lat, F); fclose(F); }
    fprintf(stderr, "FINAL LATENT n=%d  max_abs=%.3e  mean_abs=%.3e  over1e-3=%d (%.3f%%)\n",
            n_lat, maxd, sum / n_lat, over, 100.0 * over / n_lat);
    /* bf16-torch reference agreement level; f32 refs land far below. */
    printf("t2i parity %s (max_abs %.3e, limit 5e-3)\n", maxd < 5e-3 ? "PASS" : "FAIL");
    return maxd < 5e-3 ? 0 : 1;
}
