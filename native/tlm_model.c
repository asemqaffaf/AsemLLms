#define _GNU_SOURCE
#include "tlm.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "TLM1 files are little endian; big endian hosts are not supported"
#endif

#define EPS 1.0e-5

static uint32_t rd32(const unsigned char *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t rd64(const unsigned char *p) { uint64_t v; memcpy(&v, p, 8); return v; }

static void seterr(char *err, size_t n, const char *fmt, const char *a) {
    if (err && n) snprintf(err, n, fmt, a ? a : "");
}

/* ------------------------------------------------------------------ file */

typedef struct { const unsigned char *base; size_t size; uint32_t n; uint64_t table; } view;

/* Find tensor `name`; require shape rows x cols. */
static const double *find(const view *v, const char *name, uint32_t rows, uint32_t cols, char *err, size_t errlen,
                          int *bad) {
    for (uint32_t i = 0; i < v->n; i++) {
        const unsigned char *e = v->base + v->table + (uint64_t)i * TLM_ENTRY_BYTES;
        if (strncmp((const char *)e, name, TLM_NAME_MAX) != 0) continue;
        uint32_t r = rd32(e + 24), c = rd32(e + 28);
        uint64_t off = rd64(e + 32);
        if (r != rows || c != cols) { seterr(err, errlen, "tensor %s has the wrong shape", name); *bad = 1; return NULL; }
        uint64_t bytes = (uint64_t)r * c * 8u;
        if (off % 8u != 0 || off > v->size || bytes > v->size - off) {
            seterr(err, errlen, "tensor %s lies outside the file", name); *bad = 1; return NULL;
        }
        return (const double *)(const void *)(v->base + off);
    }
    seterr(err, errlen, "tensor %s is missing", name);
    *bad = 1;
    return NULL;
}

int tlm_open(const char *path, tlm_model **out, char *err, size_t errlen) {
    *out = NULL;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { seterr(err, errlen, "cannot open %s", path); return TLM_E_IO; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < (off_t)TLM_HEADER_BYTES) {
        close(fd); seterr(err, errlen, "%s is too small to be a TLM1 file", path); return TLM_E_FORMAT;
    }
    size_t size = (size_t)st.st_size;
    void *map = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { seterr(err, errlen, "cannot mmap %s", path); return TLM_E_IO; }

    const unsigned char *h = (const unsigned char *)map;
    int rc = TLM_E_FORMAT;
    tlm_model *m = NULL;
    if (memcmp(h, "TLM1", 4) != 0) { seterr(err, errlen, "%s: bad magic", path); goto fail; }
    if (rd32(h + 4) != TLM_VERSION) { seterr(err, errlen, "%s: unsupported version", path); goto fail; }
    if (rd32(h + 8) != TLM_DTYPE_F64) { seterr(err, errlen, "%s: unsupported dtype (only f64)", path); goto fail; }
    uint32_t norm = rd32(h + 12), flags = rd32(h + 16);
    uint32_t vocab = rd32(h + 20), ctx = rd32(h + 24), dim = rd32(h + 28), heads = rd32(h + 32),
             layers = rd32(h + 36), ff = rd32(h + 40), ntens = rd32(h + 44);
    uint64_t table = rd64(h + 48), data_off = rd64(h + 56), fsize = rd64(h + 64);
    const uint32_t LIM = 1u << 26;
    if (norm > TLM_NORM_RMS || vocab == 0 || ctx == 0 || dim == 0 || heads == 0 || layers == 0 || ff == 0 ||
        vocab > LIM || ctx > LIM || dim > LIM || ff > LIM || layers > 4096 || heads > dim || dim % heads != 0) {
        seterr(err, errlen, "%s: bad configuration", path); goto fail;
    }
    if (fsize != size) { seterr(err, errlen, "%s: size in header does not match the file", path); goto fail; }
    if (table < TLM_HEADER_BYTES || ntens > 1000000u ||
        table + (uint64_t)ntens * TLM_ENTRY_BYTES > size || data_off > size) {
        seterr(err, errlen, "%s: bad tensor table", path); goto fail;
    }
    view v = { h, size, ntens, table };
    m = (tlm_model *)calloc(1, sizeof(tlm_model));
    if (!m) { rc = TLM_E_MEM; goto fail; }
    m->L = (tlm_layer *)calloc(layers, sizeof(tlm_layer));
    if (!m->L) { rc = TLM_E_MEM; goto fail; }
    m->map = map; m->map_len = size;
    m->vocab = vocab; m->ctx = ctx; m->dim = dim; m->heads = heads; m->layers = layers; m->ff = ff;
    m->norm = norm; m->flags = flags;

    int bad = 0;
    int ln = norm == TLM_NORM_LAYER, bi = (flags & TLM_FLAG_BIASES) != 0;
    m->tok  = find(&v, "tok", vocab, dim, err, errlen, &bad);
    m->pos  = find(&v, "pos", ctx, dim, err, errlen, &bad);
    m->lnfg = find(&v, "lnf.g", 1, dim, err, errlen, &bad);
    m->outw = find(&v, "out.w", vocab, dim, err, errlen, &bad);
    if (ln) m->lnfb = find(&v, "lnf.b", 1, dim, err, errlen, &bad);
    if (bi) m->outb = find(&v, "out.b", 1, vocab, err, errlen, &bad);
    for (uint32_t l = 0; l < layers && !bad; l++) {
        char nm[TLM_NAME_MAX + 8];
        tlm_layer *y = &m->L[l];
#define T(field, suffix, r, c) do { snprintf(nm, sizeof nm, "l%u.%s", l, suffix); y->field = find(&v, nm, r, c, err, errlen, &bad); } while (0)
        T(g1, "ln1.g", 1, dim);
        if (ln) T(b1n, "ln1.b", 1, dim);
        T(wq, "wq", dim, dim); T(wk, "wk", dim, dim); T(wv, "wv", dim, dim); T(wo, "wo", dim, dim);
        T(g2, "ln2.g", 1, dim);
        if (ln) T(b2n, "ln2.b", 1, dim);
        T(w1, "w1", ff, dim); T(w2, "w2", dim, ff);
        if (bi) { T(b1, "b1", 1, ff); T(b2, "b2", 1, dim); }
#undef T
    }
    if (bad) goto fail;
    *out = m;
    return TLM_OK;
fail:
    if (m) { free(m->L); free(m); }
    munmap(map, size);
    return rc;
}

void tlm_close(tlm_model *m) {
    if (!m) return;
    munmap(m->map, m->map_len);
    free(m->L);
    free(m);
}

size_t tlm_param_count(const tlm_model *m) {
    size_t d = m->dim, f = m->ff, v = m->vocab, t = m->ctx, L = m->layers;
    size_t per = 4 * d * d + 2 * d * f + 2 * d;                 /* wq wk wv wo w1 w2, two norm gains */
    if (m->norm == TLM_NORM_LAYER) per += 2 * d;                /* two norm biases */
    if (m->flags & TLM_FLAG_BIASES) per += f + d;               /* b1 b2 */
    size_t n = v * d + t * d + L * per + d + v * d;             /* tok pos layers lnf.g out.w */
    if (m->norm == TLM_NORM_LAYER) n += d;
    if (m->flags & TLM_FLAG_BIASES) n += v;
    return n;
}

/* ------------------------------------------------------------------ cache */

tlm_cache *tlm_cache_new(const tlm_model *m) {
    tlm_cache *c = (tlm_cache *)calloc(1, sizeof(tlm_cache));
    if (!c) return NULL;
    size_t n = (size_t)m->layers * m->ctx * m->dim;
    c->m = m;
    c->k = (double *)calloc(n, sizeof(double));
    c->v = (double *)calloc(n, sizeof(double));
    if (!c->k || !c->v) { tlm_cache_free(c); return NULL; }
    return c;
}

tlm_cache *tlm_cache_clone(const tlm_cache *src, size_t n) {
    tlm_cache *c = tlm_cache_new(src->m);
    if (!c) return NULL;
    size_t rows = n * src->m->dim;
    for (uint32_t l = 0; l < src->m->layers; l++) {
        size_t off = (size_t)l * src->m->ctx * src->m->dim;
        memcpy(c->k + off, src->k + off, rows * sizeof(double));
        memcpy(c->v + off, src->v + off, rows * sizeof(double));
    }
    c->committed = n;
    return c;
}

void tlm_cache_free(tlm_cache *c) {
    if (!c) return;
    free(c->k);
    free(c->v);
    free(c);
}

/* ------------------------------------------------------------------ one token */

/* out[j] = (bias ? bias[j] : 0.0) + w[j,0]*x[0] + w[j,1]*x[1] + ...   (left to right, as dot/3 does) */
static void matvec(const double *w, const double *bias, const double *x, double *out, size_t rows, size_t cols) {
    for (size_t j = 0; j < rows; j++) {
        double acc = bias ? bias[j] : 0.0;
        const double *wj = w + j * cols;
        for (size_t k = 0; k < cols; k++) acc = acc + wj[k] * x[k];
        out[j] = acc;
    }
}

static void norm_row(const tlm_model *m, const double *x, const double *g, const double *b, double *out) {
    size_t d = m->dim;
    if (m->norm == TLM_NORM_LAYER) {            /* tinyllm_tf:ln_row */
        double s = 0.0;
        for (size_t j = 0; j < d; j++) s = s + x[j];
        double mean = s / (double)d, ss = 0.0;
        for (size_t j = 0; j < d; j++) { double c = x[j] - mean; ss = ss + c * c; }
        double inv = 1.0 / sqrt(ss / (double)d + EPS);
        for (size_t j = 0; j < d; j++) { double xh = (x[j] - mean) * inv; out[j] = g[j] * xh + b[j]; }
    } else {                                    /* tf.lisp:rmsnorm-fwd */
        double s = 0.0;
        for (size_t j = 0; j < d; j++) s = s + x[j] * x[j];
        double r = 1.0 / sqrt(s / (double)d + 1.0e-5);
        for (size_t j = 0; j < d; j++) out[j] = r * g[j] * x[j];
    }
}

int tlm_run_token(const tlm_model *m, tlm_cache *c, size_t n, uint32_t id, int kv_only, double *logits) {
    if (n >= m->ctx) return TLM_E_CTX;
    if (id >= m->vocab) return TLM_E_TOKEN;
    if (n > c->committed || c->m != m) return TLM_E_ARG;
    if (!kv_only && !logits) return TLM_E_ARG;

    const size_t D = m->dim, F = m->ff, H = m->heads, Dh = D / H, T = m->ctx;
    double *buf = (double *)malloc(sizeof(double) * (8 * D + F + T));
    if (!buf) return TLM_E_MEM;
    double *x = buf, *xn = x + D, *q = xn + D, *cx = q + D, *x2 = cx + D, *xn2 = x2 + D, *tmp = xn2 + D,
           *xf = tmp + D, *hact = xf + D, *sc = hact + F;
    const double scale = 1.0 / sqrt((double)Dh);

    for (size_t j = 0; j < D; j++) x[j] = m->tok[(size_t)id * D + j] + m->pos[n * D + j];

    for (uint32_t l = 0; l < m->layers; l++) {
        const tlm_layer *y = &m->L[l];
        double *Kl = c->k + (size_t)l * T * D, *Vl = c->v + (size_t)l * T * D;
        norm_row(m, x, y->g1, y->b1n, xn);
        matvec(y->wk, NULL, xn, Kl + n * D, D, D);
        matvec(y->wv, NULL, xn, Vl + n * D, D, D);
        if (kv_only && l == m->layers - 1) break;          /* nothing later can see this position's output */
        matvec(y->wq, NULL, xn, q, D, D);
        for (size_t h = 0; h < H; h++) {
            const double *qh = q + h * Dh;
            double mx = 0.0;
            for (size_t s = 0; s <= n; s++) {
                const double *ks = Kl + s * D + h * Dh;
                double a = 0.0;
                for (size_t e = 0; e < Dh; e++) a = a + qh[e] * ks[e];
                sc[s] = a * scale;
                if (s == 0 || sc[s] > mx) mx = sc[s];
            }
            double sum = 0.0;
            for (size_t s = 0; s <= n; s++) { double ex = exp(sc[s] - mx); sc[s] = ex; sum = sum + ex; }
            double *ch = cx + h * Dh;
            for (size_t e = 0; e < Dh; e++) ch[e] = 0.0;
            for (size_t s = 0; s <= n; s++) {
                double a = sc[s] / sum;
                const double *vs = Vl + s * D + h * Dh;
                for (size_t e = 0; e < Dh; e++) ch[e] = ch[e] + a * vs[e];
            }
        }
        matvec(y->wo, NULL, cx, tmp, D, D);
        for (size_t j = 0; j < D; j++) x2[j] = x[j] + tmp[j];
        norm_row(m, x2, y->g2, y->b2n, xn2);
        matvec(y->w1, y->b1, xn2, hact, F, D);
        for (size_t j = 0; j < F; j++) if (!(hact[j] > 0.0)) hact[j] = 0.0;
        matvec(y->w2, y->b2, hact, tmp, D, F);
        for (size_t j = 0; j < D; j++) x[j] = x2[j] + tmp[j];
    }
    if (!kv_only) {
        norm_row(m, x, m->lnfg, m->lnfb, xf);
        matvec(m->outw, m->outb, xf, logits, m->vocab, D);
    }
    c->committed = n + 1;
    free(buf);
    return TLM_OK;
}

int tlm_last_logits(const tlm_model *m, const int32_t *ids, size_t n, double *logits) {
    if (n == 0) return TLM_E_ARG;
    if (n > m->ctx) return TLM_E_CTX;
    tlm_cache *c = tlm_cache_new(m);
    if (!c) return TLM_E_MEM;
    int rc = TLM_OK;
    for (size_t i = 0; i < n && rc == TLM_OK; i++) {
        if (ids[i] < 0) { rc = TLM_E_TOKEN; break; }
        rc = tlm_run_token(m, c, i, (uint32_t)ids[i], i + 1 < n, i + 1 < n ? NULL : logits);
    }
    tlm_cache_free(c);
    return rc;
}

int tlm_logits_file(const char *path, const int32_t *ids, long n, double *out, long outn) {
    tlm_model *m;
    char err[160];
    int rc = tlm_open(path, &m, err, sizeof err);
    if (rc != TLM_OK) return rc;
    if (outn < (long)m->vocab || n < 0) { tlm_close(m); return TLM_E_ARG; }
    rc = tlm_last_logits(m, ids, (size_t)n, out);
    tlm_close(m);
    return rc;
}
