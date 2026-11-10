#include "tlm.h"
#include <string.h>

#if defined(TLM_BLAS_CBLAS)
#include <cblas.h>
#define TLM_HAVE_BLAS 1
#elif defined(TLM_BLAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define TLM_HAVE_BLAS 1
#endif

#ifndef TLM_BLAS_MIN_WORK
#define TLM_BLAS_MIN_WORK 262144L     /* m*k*n below this stays on the plain loops even with BLAS */
#endif

int tlm_blas_enabled(void) {
#ifdef TLM_HAVE_BLAS
    return 1;
#else
    return 0;
#endif
}

/* The arrays must not overlap (the Lisp caller passes distinct vectors); restrict lets the compiler vectorize
 * the j loops, which does not change any individual sum. */
void tlm_mm_nn(const double *restrict a, const double *restrict b, double *restrict c, long m, long k, long n, int acc) {
#ifdef TLM_HAVE_BLAS
    if (m * k * n >= TLM_BLAS_MIN_WORK) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)m, (int)n, (int)k, 1.0, a, (int)k, b, (int)n,
                    acc ? 1.0 : 0.0, c, (int)n);
        return;
    }
#endif
    if (!acc) memset(c, 0, sizeof(double) * (size_t)(m * n));
    for (long i = 0; i < m; i++) {
        const double *ai = a + i * k;
        double *ci = c + i * n;
        long p = 0;
        /* eight rows of B per pass over a row of C: each C element is loaded and stored once per eight terms,
         * but the terms are still added one after the other in order of p, as in the plain loops */
        for (; p + 8 <= k; p += 8) {
            const double a0 = ai[p], a1 = ai[p + 1], a2 = ai[p + 2], a3 = ai[p + 3],
                         a4 = ai[p + 4], a5 = ai[p + 5], a6 = ai[p + 6], a7 = ai[p + 7];
            const double *b0 = b + p * n, *b1 = b0 + n, *b2 = b1 + n, *b3 = b2 + n,
                         *b4 = b3 + n, *b5 = b4 + n, *b6 = b5 + n, *b7 = b6 + n;
            for (long j = 0; j < n; j++)
                ci[j] = ci[j] + a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j]
                              + a4 * b4[j] + a5 * b5[j] + a6 * b6[j] + a7 * b7[j];
        }
        for (; p < k; p++) {
            const double ap = ai[p];
            const double *bp = b + p * n;
            for (long j = 0; j < n; j++) ci[j] = ci[j] + ap * bp[j];
        }
    }
}

void tlm_mm_nt(const double *restrict a, const double *restrict b, double *restrict c, long m, long k, long n, int acc) {
#ifdef TLM_HAVE_BLAS
    if (m * k * n >= TLM_BLAS_MIN_WORK) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (int)m, (int)n, (int)k, 1.0, a, (int)k, b, (int)k,
                    acc ? 1.0 : 0.0, c, (int)n);
        return;
    }
#endif
    if (!acc) memset(c, 0, sizeof(double) * (size_t)(m * n));
    for (long i = 0; i < m; i++) {
        const double *ai = a + i * k;
        long j = 0;
        /* four output columns at a time: four independent sums, each still added in order of p */
        for (; j + 4 <= n; j += 4) {
            const double *b0 = b + j * k, *b1 = b0 + k, *b2 = b1 + k, *b3 = b2 + k;
            double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
            for (long p = 0; p < k; p++) {
                double ap = ai[p];
                s0 = s0 + ap * b0[p]; s1 = s1 + ap * b1[p]; s2 = s2 + ap * b2[p]; s3 = s3 + ap * b3[p];
            }
            double *ci = c + i * n + j;
            ci[0] = ci[0] + s0; ci[1] = ci[1] + s1; ci[2] = ci[2] + s2; ci[3] = ci[3] + s3;
        }
        for (; j < n; j++) {
            double s = 0.0;
            const double *bj = b + j * k;
            for (long p = 0; p < k; p++) s = s + ai[p] * bj[p];
            c[i * n + j] = c[i * n + j] + s;
        }
    }
}

void tlm_mm_tn(const double *restrict a, const double *restrict b, double *restrict c, long m, long k, long n, int acc) {
#ifdef TLM_HAVE_BLAS
    if (m * k * n >= TLM_BLAS_MIN_WORK) {
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, (int)k, (int)n, (int)m, 1.0, a, (int)k, b, (int)n,
                    acc ? 1.0 : 0.0, c, (int)n);
        return;
    }
#endif
    if (!acc) memset(c, 0, sizeof(double) * (size_t)(k * n));
    long i = 0;
    for (; i + 8 <= m; i += 8) {                 /* eight rows of A and B per pass, added in order of i */
        const double *b0 = b + i * n, *b1 = b0 + n, *b2 = b1 + n, *b3 = b2 + n,
                     *b4 = b3 + n, *b5 = b4 + n, *b6 = b5 + n, *b7 = b6 + n;
        for (long p = 0; p < k; p++) {
            const double a0 = a[i * k + p], a1 = a[(i + 1) * k + p], a2 = a[(i + 2) * k + p], a3 = a[(i + 3) * k + p],
                         a4 = a[(i + 4) * k + p], a5 = a[(i + 5) * k + p], a6 = a[(i + 6) * k + p], a7 = a[(i + 7) * k + p];
            double *cp = c + p * n;
            for (long j = 0; j < n; j++)
                cp[j] = cp[j] + a0 * b0[j] + a1 * b1[j] + a2 * b2[j] + a3 * b3[j]
                              + a4 * b4[j] + a5 * b5[j] + a6 * b6[j] + a7 * b7[j];
        }
    }
    for (; i < m; i++)
        for (long p = 0; p < k; p++) {
            const double aip = a[i * k + p];
            const double *bi = b + i * n;
            double *cp = c + p * n;
            for (long j = 0; j < n; j++) cp[j] = cp[j] + aip * bi[j];
        }
}
