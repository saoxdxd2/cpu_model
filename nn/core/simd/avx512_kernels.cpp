// ============================================================================
// NCA — AVX-512 Math Kernels
// core/simd/avx512_kernels.cpp
//
// Maximum Optimization:
// - Software prefetching (_mm_prefetch)
// - Fast Hardware RSQRT (_mm512_rsqrt14_ps) + Newton Raphson
// - __restrict pointers for no-alias guarantee
// - C++20 [[likely]] attributes
// ============================================================================

#include "core/simd/avx512_kernels.hpp"
#include "core/simd/avx512_math.hpp"
#include <immintrin.h>

namespace nca::simd::avx512 {

void rmsnorm(float* __restrict out, const float* __restrict in, const float* __restrict weight, size_t size, float eps) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    
    const float *p_in = in, *p_w = weight;
    float *p_out = out;
    size_t rem = size;

    // Pass 1: Accumulate Sum of Squares
    for (; rem >= 64; rem -= 64, p_in += 64) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_in + 128), _MM_HINT_T0);
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(p_in),      _mm512_loadu_ps(p_in),      s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(p_in + 16), _mm512_loadu_ps(p_in + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_loadu_ps(p_in + 32), _mm512_loadu_ps(p_in + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_loadu_ps(p_in + 48), _mm512_loadu_ps(p_in + 48), s3);
    }
    s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));

    for (; rem >= 16; rem -= 16, p_in += 16) [[unlikely]] 
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(p_in), _mm512_loadu_ps(p_in), s0);
    
    if (rem > 0) [[unlikely]] {
        __mmask16 mask = (1U << rem) - 1;
        __m512 x_t = _mm512_maskz_loadu_ps(mask, p_in);
        s0 = _mm512_fmadd_ps(x_t, x_t, s0);
    }

    // Hardware RSQRT (14-bit precision) + Newton Raphson step (23-bit precision)
    // Avoids massive latency of std::sqrt scalar division.
    float sum_val = _mm512_reduce_add_ps(s0) / size + eps;
    __m512 v_x = _mm512_set1_ps(sum_val);
    __m512 v_y0 = _mm512_rsqrt14_ps(v_x); 
    __m512 v_half_x = _mm512_mul_ps(v_x, _mm512_set1_ps(0.5f));
    __m512 v_y0_sq = _mm512_mul_ps(v_y0, v_y0);
    // y1 = y0 * (1.5 - 0.5 * x * y0^2)
    __m512 v_inv = _mm512_mul_ps(v_y0, _mm512_fnmadd_ps(v_half_x, v_y0_sq, _mm512_set1_ps(1.5f)));
    
    // Pass 2: Normalize & Weight
    p_in = in; rem = size;

    for (; rem >= 64; rem -= 64, p_in += 64, p_w += 64, p_out += 64) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_in + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_w + 128), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_out + 128), _MM_HINT_T0);

        _mm512_storeu_ps(p_out,      _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(p_in),      v_inv), _mm512_loadu_ps(p_w)));
        _mm512_storeu_ps(p_out + 16, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(p_in + 16), v_inv), _mm512_loadu_ps(p_w + 16)));
        _mm512_storeu_ps(p_out + 32, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(p_in + 32), v_inv), _mm512_loadu_ps(p_w + 32)));
        _mm512_storeu_ps(p_out + 48, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(p_in + 48), v_inv), _mm512_loadu_ps(p_w + 48)));
    }

    for (; rem >= 16; rem -= 16, p_in += 16, p_w += 16, p_out += 16) [[unlikely]]
        _mm512_storeu_ps(p_out, _mm512_mul_ps(_mm512_mul_ps(_mm512_loadu_ps(p_in), v_inv), _mm512_loadu_ps(p_w)));

    if (rem > 0) [[unlikely]] {
        __mmask16 mask = (1U << rem) - 1;
        _mm512_mask_storeu_ps(p_out, mask, _mm512_mul_ps(_mm512_mul_ps(_mm512_maskz_loadu_ps(mask, p_in), v_inv), _mm512_maskz_loadu_ps(mask, p_w)));
    }
}

void silu(float* __restrict data, size_t size) {
    float* p_data = data;
    size_t rem = size;

    for (; rem >= 64; rem -= 64, p_data += 64) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_data + 128), _MM_HINT_T0);
        
        _mm512_storeu_ps(p_data,      silu_ps(_mm512_loadu_ps(p_data)));
        _mm512_storeu_ps(p_data + 16, silu_ps(_mm512_loadu_ps(p_data + 16)));
        _mm512_storeu_ps(p_data + 32, silu_ps(_mm512_loadu_ps(p_data + 32)));
        _mm512_storeu_ps(p_data + 48, silu_ps(_mm512_loadu_ps(p_data + 48)));
    }

    for (; rem >= 16; rem -= 16, p_data += 16) [[unlikely]]
        _mm512_storeu_ps(p_data, silu_ps(_mm512_loadu_ps(p_data)));

    if (rem > 0) [[unlikely]] {
        __mmask16 mask = (1U << rem) - 1;
        _mm512_mask_storeu_ps(p_data, mask, silu_ps(_mm512_maskz_loadu_ps(mask, p_data)));
    }
}

} // namespace nca::simd::avx512
