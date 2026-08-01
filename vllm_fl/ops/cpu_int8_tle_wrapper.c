/* W8A8 backing function for the FlagTree TLE CPU GEMM op.
 *
 * Registered via neon.register_c_function as the extern symbol
 * `sdot_gemm_q4_0_v2_smmla_bf16` (the compiled TLE op passes raw pointers
 * plus m/k/n, so the backing implementation defines the weight layout).
 * Here rhs is a KleidiAI qsi8cxp4x8 packed int8 weight blob: decode (m==1)
 * runs the 1x4 NEON dotprod GEMV, prefill (m>1) the 16x4 NEON i8mm GEMM.
 * Kernels emit f32; rows are converted to bf16 in the writing thread.
 * KleidiAI ukernel symbols are resolved from the process-global
 * libkai_w8a8.so loaded by cpu_int8_tleraw.py.
 */
#if defined(__aarch64__)
#include <float.h>
#include <omp.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define _CAT(a, b) a##b
#define CAT(a, b) _CAT(a, b)

#define GEMV matmul_clamp_f32_qai8dxp1x8_qsi8cxp4x8_1x4_neon_dotprod
#define GEMM matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm
#define VGET(f) CAT(CAT(kai_get_, f), CAT(_, GEMV))
#define VRun CAT(kai_run_, GEMV)
#define MGET(f) CAT(CAT(kai_get_, f), CAT(_, GEMM))
#define MRun CAT(kai_run_, GEMM)

size_t VGET(mr)(void);
size_t VGET(kr)(void);
size_t VGET(sr)(void);
size_t VGET(nr)(void);
size_t VGET(n_step)(void);
size_t VGET(rhs_packed_offset)(size_t n_idx, size_t k);
void VRun(size_t m, size_t n, size_t k,
          const void *lhs_packed, const void *rhs_packed, float *dst,
          size_t dst_stride_row, size_t dst_stride_col,
          float scalar_min, float scalar_max);

size_t MGET(mr)(void);
size_t MGET(kr)(void);
size_t MGET(sr)(void);
size_t MGET(nr)(void);
size_t MGET(n_step)(void);
size_t MGET(rhs_packed_offset)(size_t n_idx, size_t k);
void MRun(size_t m, size_t n, size_t k,
          const void *lhs_packed, const void *rhs_packed, float *dst,
          size_t dst_stride_row, size_t dst_stride_col,
          float scalar_min, float scalar_max);

size_t kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr);
size_t kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(
    size_t m_idx, size_t k, size_t mr, size_t kr, size_t sr);
void kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
    size_t m, size_t k, size_t mr, size_t kr, size_t sr,
    size_t m_idx_start, const void *lhs, size_t lhs_stride,
    void *lhs_packed);

static _Thread_local uint8_t *w8a8_lhs_scratch;
static _Thread_local size_t w8a8_lhs_capacity;
static _Thread_local float *w8a8_dst_scratch;
static _Thread_local size_t w8a8_dst_capacity;

static void *reserve(uint8_t **buf, size_t *cap, size_t size) {
    if (size <= *cap) {
        return *buf;
    }
    void *next = realloc(*buf, size);
    if (next == NULL) {
        abort();
    }
    *buf = (uint8_t *)next;
    *cap = size;
    return next;
}

static inline void f32_to_bf16_row(const float *src, uint16_t *dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        memcpy(&bits, &src[i], sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u &&
            (bits & 0x007fffffu)) {
            dst[i] = (uint16_t)((bits >> 16) | 0x0040u);
            continue;
        }
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        dst[i] = (uint16_t)(bits >> 16);
    }
}

static void w8a8_run_gemv(const uint16_t *x, const void *rhs, uint16_t *out,
                          size_t n, size_t k) {
    const size_t mr = VGET(mr)();
    const size_t kr = VGET(kr)();
    const size_t sr = VGET(sr)();
    void *lhs = reserve(&w8a8_lhs_scratch, &w8a8_lhs_capacity,
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(1, k, mr, kr, sr));
    kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
        1, k, mr, kr, sr, 0, x, k * sizeof(uint16_t), lhs);
    float *dst = (float *)reserve((uint8_t **)&w8a8_dst_scratch,
                                  &w8a8_dst_capacity, n * sizeof(float));

    const size_t nr = VGET(nr)();
    size_t n_step = VGET(n_step)();
    if (n_step < nr) {
        n_step = nr;
    }
    const size_t n_tiles = (n + n_step - 1) / n_step;
#pragma omp parallel
    {
        const size_t nthreads = (size_t)omp_get_num_threads();
        const size_t tid = (size_t)omp_get_thread_num();
        const size_t per = (n_tiles + nthreads - 1) / nthreads;
        const size_t begin = tid * per;
        size_t end = begin + per;
        if (end > n_tiles) {
            end = n_tiles;
        }
        for (size_t t = begin; t < end; ++t) {
            const size_t n0 = t * n_step;
            const size_t width = n0 + n_step <= n ? n_step : n - n0;
            const uint8_t *rhs_t =
                (const uint8_t *)rhs + VGET(rhs_packed_offset)(n0, k);
            VRun(1, width, k, lhs, rhs_t, dst + n0, n * sizeof(float),
                 sizeof(float), -FLT_MAX, FLT_MAX);
            f32_to_bf16_row(dst + n0, out + n0, width);
        }
    }
}

static void w8a8_run_gemm(const uint16_t *x, const void *rhs, uint16_t *out,
                          size_t m, size_t n, size_t k) {
    const size_t mr = MGET(mr)();
    const size_t kr = MGET(kr)();
    const size_t sr = MGET(sr)();
    const size_t nr = MGET(nr)();
    size_t n_step = MGET(n_step)();
    if (n_step < nr) {
        n_step = nr;
    }
    void *lhs = reserve(&w8a8_lhs_scratch, &w8a8_lhs_capacity,
        kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_bf16_neon(m, k, mr, kr, sr));
    float *dst = (float *)reserve((uint8_t **)&w8a8_dst_scratch,
                                  &w8a8_dst_capacity, m * n * sizeof(float));
    const size_t m_tiles = (m + mr - 1) / mr;
    const size_t n_tiles = (n + n_step - 1) / n_step;

#pragma omp parallel
    {
        const size_t nthreads = (size_t)omp_get_num_threads();
        const size_t tid = (size_t)omp_get_thread_num();

        const size_t m_per = (m_tiles + nthreads - 1) / nthreads;
        const size_t mt0 = tid * m_per;
        size_t mt1 = mt0 + m_per;
        if (mt1 > m_tiles) {
            mt1 = m_tiles;
        }
        const size_t m0 = mt0 * mr;
        if (m0 < m) {
            const size_t rows = mt1 * mr <= m ? (mt1 - mt0) * mr : m - m0;
            uint8_t *lhs_t = (uint8_t *)lhs +
                kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_bf16_neon(
                    m0, k, mr, kr, sr);
            kai_run_lhs_quant_pack_qai8dxp_bf16_neon(
                rows, k, mr, kr, sr, 0, x + m0 * k, k * sizeof(uint16_t), lhs_t);
        }
#pragma omp barrier

        const size_t n_per = (n_tiles + nthreads - 1) / nthreads;
        const size_t nt0 = tid * n_per;
        size_t nt1 = nt0 + n_per;
        if (nt1 > n_tiles) {
            nt1 = n_tiles;
        }
        for (size_t t = nt0; t < nt1; ++t) {
            const size_t n0 = t * n_step;
            const size_t width = n0 + n_step <= n ? n_step : n - n0;
            const uint8_t *rhs_t =
                (const uint8_t *)rhs + MGET(rhs_packed_offset)(n0, k);
            MRun(m, width, k, lhs, rhs_t, dst + n0, n * sizeof(float),
                 sizeof(float), -FLT_MAX, FLT_MAX);
            for (size_t r = 0; r < m; ++r) {
                f32_to_bf16_row(dst + r * n + n0, out + r * n + n0, width);
            }
        }
    }
}

void sdot_gemm_q4_0_v2_smmla_bf16(
    const uint16_t *x, const void *rhs, uint16_t *out,
    int64_t m_value, int64_t k_value, int64_t n_value) {
    const size_t m = (size_t)m_value;
    const size_t k = (size_t)k_value;
    const size_t n = (size_t)n_value;
    if (m == 1) {
        w8a8_run_gemv(x, rhs, out, n, k);
    } else {
        w8a8_run_gemm(x, rhs, out, m, n, k);
    }
}
#endif
