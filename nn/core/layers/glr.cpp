// ============================================================================
// NCA — GLR Backbone Layer
// core/layers/glr.cpp
// ============================================================================

#include "core/layers/glr.hpp"
#include "core/simd/dispatch.hpp"
#include <immintrin.h>

namespace nca::layers {

void glr_step_avx512(float* __restrict h, const float* __restrict alpha, const float* __restrict beta, const float* __restrict x, size_t d_size) {
    float* p_h = h;
    const float* p_a = alpha;
    const float* p_b = beta;
    const float* p_x = x;
    size_t rem = d_size;

    for (; rem >= 64; rem -= 64, p_h += 64, p_a += 64, p_b += 64, p_x += 64) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_h + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_a + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_b + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_x + 128), _MM_HINT_T0);

        __m512 v_bx0 = _mm512_mul_ps(_mm512_loadu_ps(p_b),      _mm512_loadu_ps(p_x));
        __m512 v_bx1 = _mm512_mul_ps(_mm512_loadu_ps(p_b + 16), _mm512_loadu_ps(p_x + 16));
        __m512 v_bx2 = _mm512_mul_ps(_mm512_loadu_ps(p_b + 32), _mm512_loadu_ps(p_x + 32));
        __m512 v_bx3 = _mm512_mul_ps(_mm512_loadu_ps(p_b + 48), _mm512_loadu_ps(p_x + 48));

        __m512 res0 = _mm512_fmadd_ps(_mm512_loadu_ps(p_a),      _mm512_loadu_ps(p_h),      v_bx0);
        __m512 res1 = _mm512_fmadd_ps(_mm512_loadu_ps(p_a + 16), _mm512_loadu_ps(p_h + 16), v_bx1);
        __m512 res2 = _mm512_fmadd_ps(_mm512_loadu_ps(p_a + 32), _mm512_loadu_ps(p_h + 32), v_bx2);
        __m512 res3 = _mm512_fmadd_ps(_mm512_loadu_ps(p_a + 48), _mm512_loadu_ps(p_h + 48), v_bx3);

        _mm512_storeu_ps(p_h,      res0);
        _mm512_storeu_ps(p_h + 16, res1);
        _mm512_storeu_ps(p_h + 32, res2);
        _mm512_storeu_ps(p_h + 48, res3);
    }
}

void glr_step_avx2(float* __restrict h, const float* __restrict alpha, const float* __restrict beta, const float* __restrict x, size_t d_size) {
    float* p_h = h;
    const float* p_a = alpha;
    const float* p_b = beta;
    const float* p_x = x;
    size_t rem = d_size;

    for (; rem >= 32; rem -= 32, p_h += 32, p_a += 32, p_b += 32, p_x += 32) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_h + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_a + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_b + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_x + 64), _MM_HINT_T0);

        __m256 v_bx0 = _mm256_mul_ps(_mm256_loadu_ps(p_b),      _mm256_loadu_ps(p_x));
        __m256 v_bx1 = _mm256_mul_ps(_mm256_loadu_ps(p_b + 8),  _mm256_loadu_ps(p_x + 8));
        __m256 v_bx2 = _mm256_mul_ps(_mm256_loadu_ps(p_b + 16), _mm256_loadu_ps(p_x + 16));
        __m256 v_bx3 = _mm256_mul_ps(_mm256_loadu_ps(p_b + 24), _mm256_loadu_ps(p_x + 24));

        __m256 res0 = _mm256_fmadd_ps(_mm256_loadu_ps(p_a),      _mm256_loadu_ps(p_h),      v_bx0);
        __m256 res1 = _mm256_fmadd_ps(_mm256_loadu_ps(p_a + 8),  _mm256_loadu_ps(p_h + 8),  v_bx1);
        __m256 res2 = _mm256_fmadd_ps(_mm256_loadu_ps(p_a + 16), _mm256_loadu_ps(p_h + 16), v_bx2);
        __m256 res3 = _mm256_fmadd_ps(_mm256_loadu_ps(p_a + 24), _mm256_loadu_ps(p_h + 24), v_bx3);

        _mm256_storeu_ps(p_h,      res0);
        _mm256_storeu_ps(p_h + 8,  res1);
        _mm256_storeu_ps(p_h + 16, res2);
        _mm256_storeu_ps(p_h + 24, res3);
    }
}

void glr_step_scalar(float* __restrict h, const float* __restrict alpha, const float* __restrict beta, const float* __restrict x, size_t d_size) {
    for (size_t i = 0; i < d_size; ++i) [[likely]] {
        h[i] = alpha[i] * h[i] + beta[i] * x[i];
    }
}

void glr_step(float* h, const float* alpha, const float* beta, const float* x, size_t d_size) {
    NCA_DISPATCH_KERNEL(glr_step_avx512, glr_step_avx2, glr_step_scalar, h, alpha, beta, x, d_size);
}

} // namespace nca::layers
