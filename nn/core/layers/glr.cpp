// ============================================================================
// NCA -- GLR Backbone Layer
// core/layers/glr.cpp
//
// Uses compile-time CachePolicy to select optimal strategy.
// GLR working set: 4 arrays * D * 4 bytes.
//   D=2048 → 32KB → L1_HOT (no prefetch, tight loop)
//   D=8192 → 128KB → L2_STREAM (prefetch to L1, regular stores)
// ============================================================================

#include "core/layers/glr.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/cache_policy.hpp"
#include <immintrin.h>

namespace nca::layers {

// Compile-time: D=8192 → 4*8192*4 = 128KB → L2_STREAM
using Policy = nca::simd::CachePolicy<4 * 8192 * 4>;

void glr_step_avx512(float* __restrict h, const float* __restrict alpha, const float* __restrict beta, const float* __restrict x, size_t d_size) {
    float* p_h = h;
    const float* p_a = alpha;
    const float* p_b = beta;
    const float* p_x = x;
    size_t rem = d_size;

    // Policy::prefetch_dist == 4 cache lines = 256 bytes ahead
    constexpr size_t PF = Policy::prefetch_dist * 64 / sizeof(float);

    for (; rem >= 64; rem -= 64, p_h += 64, p_a += 64, p_b += 64, p_x += 64) [[likely]] {
        // Branchless prefetch: distance chosen at compile time by CachePolicy
        if constexpr (Policy::prefetch_dist > 0) {
            _mm_prefetch(reinterpret_cast<const char*>(p_a + PF), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p_b + PF), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(p_x + PF), _MM_HINT_T0);
        }

        __m512 v_bx0 = _mm512_mul_ps(_mm512_loadu_ps(p_b),      _mm512_loadu_ps(p_x));
        __m512 v_bx1 = _mm512_mul_ps(_mm512_loadu_ps(p_b + 16), _mm512_loadu_ps(p_x + 16));
        __m512 v_bx2 = _mm512_mul_ps(_mm512_loadu_ps(p_b + 32), _mm512_loadu_ps(p_x + 32));
        __m512 v_bx3 = _mm512_mul_ps(_mm512_loadu_ps(p_b + 48), _mm512_loadu_ps(p_x + 48));

        __m512 res0 = _mm512_fmadd_ps(_mm512_loadu_ps(p_a),      _mm512_loadu_ps(p_h),      v_bx0);
        __m512 res1 = _mm512_fmadd_ps(_mm512_loadu_ps(p_a + 16), _mm512_loadu_ps(p_h + 16), v_bx1);
        __m512 res2 = _mm512_fmadd_ps(_mm512_loadu_ps(p_a + 32), _mm512_loadu_ps(p_h + 32), v_bx2);
        __m512 res3 = _mm512_fmadd_ps(_mm512_loadu_ps(p_a + 48), _mm512_loadu_ps(p_h + 48), v_bx3);

        // Store policy: constexpr if decides at compile time. Zero runtime branches.
        if constexpr (Policy::use_nt_stores) {
            _mm512_stream_ps(p_h,      res0);
            _mm512_stream_ps(p_h + 16, res1);
            _mm512_stream_ps(p_h + 32, res2);
            _mm512_stream_ps(p_h + 48, res3);
        } else {
            _mm512_storeu_ps(p_h,      res0);
            _mm512_storeu_ps(p_h + 16, res1);
            _mm512_storeu_ps(p_h + 32, res2);
            _mm512_storeu_ps(p_h + 48, res3);
        }
    }
    if constexpr (Policy::use_nt_stores) _mm_sfence();

    // Branchless tail: masked operations instead of if/for
    if (rem > 0) {
        __mmask16 m0 = nca::simd::tail_mask(rem);
        __mmask16 m1 = nca::simd::tail_mask(rem > 16 ? rem - 16 : 0);
        __mmask16 m2 = nca::simd::tail_mask(rem > 32 ? rem - 32 : 0);
        __mmask16 m3 = nca::simd::tail_mask(rem > 48 ? rem - 48 : 0);

        __m512 v_bx0 = _mm512_mul_ps(_mm512_maskz_loadu_ps(m0, p_b),      _mm512_maskz_loadu_ps(m0, p_x));
        __m512 v_bx1 = _mm512_mul_ps(_mm512_maskz_loadu_ps(m1, p_b + 16), _mm512_maskz_loadu_ps(m1, p_x + 16));
        __m512 v_bx2 = _mm512_mul_ps(_mm512_maskz_loadu_ps(m2, p_b + 32), _mm512_maskz_loadu_ps(m2, p_x + 32));
        __m512 v_bx3 = _mm512_mul_ps(_mm512_maskz_loadu_ps(m3, p_b + 48), _mm512_maskz_loadu_ps(m3, p_x + 48));

        __m512 res0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m0, p_a),      _mm512_maskz_loadu_ps(m0, p_h),      v_bx0);
        __m512 res1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m1, p_a + 16), _mm512_maskz_loadu_ps(m1, p_h + 16), v_bx1);
        __m512 res2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m2, p_a + 32), _mm512_maskz_loadu_ps(m2, p_h + 32), v_bx2);
        __m512 res3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m3, p_a + 48), _mm512_maskz_loadu_ps(m3, p_h + 48), v_bx3);

        _mm512_mask_storeu_ps(p_h,      m0, res0);
        _mm512_mask_storeu_ps(p_h + 16, m1, res1);
        _mm512_mask_storeu_ps(p_h + 32, m2, res2);
        _mm512_mask_storeu_ps(p_h + 48, m3, res3);
    }
}

void glr_step_avx2(float* __restrict h, const float* __restrict alpha, const float* __restrict beta, const float* __restrict x, size_t d_size) {
    float* p_h = h;
    const float* p_a = alpha;
    const float* p_b = beta;
    const float* p_x = x;
    size_t rem = d_size;

    for (; rem >= 32; rem -= 32, p_h += 32, p_a += 32, p_b += 32, p_x += 32) [[likely]] {
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
