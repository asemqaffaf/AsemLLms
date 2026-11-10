/* TLM1: page-aligned flat weight files and the C inference/matmul library shared by the Erlang NIF and
 * the Lisp (sb-alien) side of AsemLLM. No Erlang or Lisp headers in here; see FORMAT.md for the file layout.
 *
 * Numerics: everything is IEEE double and every sum is accumulated left to right in the same order as
 * tinyllm_tf.erl and lisp/src/tf.lisp, so results are bit-identical to those implementations. Build with
 * -ffp-contract=off (the Makefile does): a fused multiply-add would change the last bit on arm64.
 * The one exception is an optional BLAS build (make BLAS=...), which is faster and NOT bit-identical.
 */
#ifndef TLM_H
#define TLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TLM_VERSION      1u
#define TLM_PAGE         4096u
#define TLM_TENSOR_ALIGN 256u
#define TLM_NAME_MAX     24u
#define TLM_HEADER_BYTES 128u
#define TLM_ENTRY_BYTES  40u

enum { TLM_DTYPE_F64 = 1 };
enum { TLM_NORM_LAYER = 0, TLM_NORM_RMS = 1 };
enum { TLM_FLAG_BIASES = 1 };

enum { TLM_OK = 0, TLM_E_CTX = 1, TLM_E_TOKEN = 2, TLM_E_MEM = 3, TLM_E_IO = 4, TLM_E_FORMAT = 5, TLM_E_ARG = 6 };

typedef struct {
    const double *g1, *b1n, *wq, *wk, *wv, *wo, *g2, *b2n, *w1, *b1, *w2, *b2;
} tlm_layer;

typedef struct tlm_model {
    void *map;                 /* the whole file, mmap'd read-only (page aligned) */
    size_t map_len;
    uint32_t vocab, ctx, dim, heads, layers, ff, norm, flags;
    const double *tok, *pos, *lnfg, *lnfb, *outw, *outb;   /* b* are NULL when the file has none */
    tlm_layer *L;
} tlm_model;

/* KV cache: per layer, ctx rows of dim values (head h owns columns [h*dh, (h+1)*dh)). Positions >= committed
 * are garbage. A cache value seen by a caller is (cache, n): see tlm_cache_clone for the copy-on-write rule. */
typedef struct tlm_cache {
    const tlm_model *m;
    size_t committed;
    double *k, *v;
} tlm_cache;

/* ---- model file */
int  tlm_open(const char *path, tlm_model **out, char *err, size_t errlen);
void tlm_close(tlm_model *m);
size_t tlm_param_count(const tlm_model *m);

/* ---- incremental decoding */
tlm_cache *tlm_cache_new(const tlm_model *m);
tlm_cache *tlm_cache_clone(const tlm_cache *src, size_t n);      /* copies the first n positions */
void       tlm_cache_free(tlm_cache *c);
/* Feed token id at position n (0-based; n <= committed). kv_only: only extend the cache (skips the output
 * projection and, in the last layer, everything except K and V). logits (vocab doubles) may be NULL if kv_only. */
int tlm_run_token(const tlm_model *m, tlm_cache *c, size_t n, uint32_t id, int kv_only, double *logits);

/* Convenience for tests and other languages: logits for the token after ids[0..n). */
int tlm_last_logits(const tlm_model *m, const int32_t *ids, size_t n, double *logits);
int tlm_logits_file(const char *path, const int32_t *ids, long n, double *out, long outn);

/* ---- matrix kernels (row-major doubles), same contracts as mm-nn / mm-nt / mm-tn in lisp/src/tf.lisp:
 *   nn: C(m x n) = [C +] A(m x k) B(k x n)      nt: C(m x n) = [C +] A(m x k) B(n x k)^T
 *   tn: C(k x n) = [C +] A(m x k)^T B(m x n)    acc != 0 adds into C, otherwise C is overwritten. */
void tlm_mm_nn(const double *a, const double *b, double *c, long m, long k, long n, int acc);
void tlm_mm_nt(const double *a, const double *b, double *c, long m, long k, long n, int acc);
void tlm_mm_tn(const double *a, const double *b, double *c, long m, long k, long n, int acc);
int  tlm_blas_enabled(void);       /* 1 if built with BLAS (then big products are not bit-identical) */

#ifdef __cplusplus
}
#endif
#endif
