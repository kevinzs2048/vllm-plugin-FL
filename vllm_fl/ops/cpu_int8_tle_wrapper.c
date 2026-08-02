/* Thin ABI adapter for the FlagTree TLE CPU GEMM op.
 *
 * cpu_int8_tleraw.py loads libkai_w8a8.so with RTLD_GLOBAL before the
 * generated kernel is loaded. Keep the W8A8 implementation in that library;
 * this source only maps the compiler's legacy q4 symbol and argument order to
 * the shared implementation.
 */
#if defined(__aarch64__)
#include <stddef.h>
#include <stdint.h>

void fl_w8a8_linear(size_t m, size_t n, size_t k,
                    const uint16_t *x_bf16, const void *rhs_packed,
                    uint16_t *out_bf16);

void sdot_gemm_q4_0_v2_smmla_bf16(
    const uint16_t *x, const void *rhs, uint16_t *out,
    int64_t m_value, int64_t k_value, int64_t n_value) {
    fl_w8a8_linear((size_t)m_value, (size_t)n_value, (size_t)k_value,
                   x, rhs, out);
}
#endif
