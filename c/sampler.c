#include "sampler.h"
#include "tensor.h"
#include "util.h"
#include <math.h>
#include <float.h>

void rng_seed(Rng *r, uint64_t seed) {
    r->s = seed ? seed : 0x9e3779b97f4a7c15ULL;
}

static uint64_t rng_u64(Rng *r) {
    uint64_t x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

double rng_f64(Rng *r) {
    return (double)(rng_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}

void softmax_last(const float *logits, int n, double *probs) {
    double m = -INFINITY;
    for (int i = 0; i < n; i++) {
        double v = (double)logits[i];
        if (v > m) m = v;
    }
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double e = exp((double)logits[i] - m);
        probs[i] = e;
        sum += e;
    }
    double inv = sum > 0.0 ? 1.0 / sum : 0.0;
    for (int i = 0; i < n; i++) probs[i] *= inv;
}

typedef struct {
    double v;
    int i;
} Pair;

static int pair_cmp_desc(const void *a, const void *b) {
    const Pair *pa = a, *pb = b;
    if (pa->v < pb->v) return 1;
    if (pa->v > pb->v) return -1;
    return pa->i - pb->i;
}

int sample_token(const float *logits, int n, float temperature, int top_k, float top_p, Rng *rng) {
    if (temperature <= 0.0f || top_k == 1) return argmax_f32(logits, n);

    double *x = xmalloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) x[i] = (double)logits[i] / (double)temperature;

    if (top_k > 0 && top_k < n) {
        Pair *ps = xmalloc((size_t)n * sizeof(Pair));
        for (int i = 0; i < n; i++) {
            ps[i].v = x[i];
            ps[i].i = i;
        }
        qsort(ps, (size_t)n, sizeof(Pair), pair_cmp_desc);
        double thresh = ps[top_k - 1].v;
        free(ps);
        for (int i = 0; i < n; i++) {
            if (x[i] < thresh) x[i] = -INFINITY;
        }
    }

    double *probs = xmalloc((size_t)n * sizeof(double));
    double m = -INFINITY;
    for (int i = 0; i < n; i++)
        if (x[i] > m) m = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double e = isfinite(x[i]) ? exp(x[i] - m) : 0.0;
        probs[i] = e;
        sum += e;
    }
    if (sum <= 0.0) {
        int best = argmax_f32(logits, n);
        free(x);
        free(probs);
        return best;
    }
    for (int i = 0; i < n; i++) probs[i] /= sum;

    if (top_p < 1.0f) {
        Pair *ps = xmalloc((size_t)n * sizeof(Pair));
        for (int i = 0; i < n; i++) {
            ps[i].v = probs[i];
            ps[i].i = i;
        }
        qsort(ps, (size_t)n, sizeof(Pair), pair_cmp_desc);
        char *keep = xcalloc((size_t)n, 1);
        double cdf = 0.0;
        int nkeep = 0;
        for (int i = 0; i < n; i++) {
            cdf += ps[i].v;
            if (cdf <= (double)top_p || i == 0) {
                keep[ps[i].i] = 1;
                nkeep++;
            } else {
                keep[ps[i].i] = 1; /* token that just crosses top_p */
                nkeep++;
                break;
            }
        }
        free(ps);
        double z = 0.0;
        for (int i = 0; i < n; i++) {
            if (!keep[i]) probs[i] = 0.0;
            else z += probs[i];
        }
        free(keep);
        if (z <= 0.0) {
            int best = 0;
            for (int i = 1; i < n; i++)
                if (x[i] > x[best]) best = i;
            free(x);
            free(probs);
            return best;
        }
        for (int i = 0; i < n; i++) probs[i] /= z;
    }

    Rng local;
    if (!rng) {
        rng_seed(&local, 0xC0FFEEULL);
        rng = &local;
    }
    double u = rng_f64(rng);
    double acc = 0.0;
    int chosen = n - 1;
    for (int i = 0; i < n; i++) {
        acc += probs[i];
        if (u < acc) {
            chosen = i;
            break;
        }
    }
    free(x);
    free(probs);
    return chosen;
}
