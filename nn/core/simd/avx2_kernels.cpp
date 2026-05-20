// ============================================================================
// NCA — AVX2 Math Kernels
// core/simd/avx2_kernels.cpp
// ============================================================================

#include "core/simd/avx2_kernels.hpp"
#include "core/simd/avx2_math.hpp"
#include <immintrin.h>
#include <cmath>

namespace nca::simd::avx2 {

void rmsnorm(float* __restrict out, const float* __restrict in, const float* __restrict weight, size_t size, float eps) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    const float *p_in = in, *p_w = weight;
    float *p_out = out;
    size_t rem = size;

    for (; rem >= 32; rem -= 32, p_in += 32) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_in + 64), _MM_HINT_T0);
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in),      _mm256_loadu_ps(p_in),      s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in + 8),  _mm256_loadu_ps(p_in + 8),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in + 16), _mm256_loadu_ps(p_in + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in + 24), _mm256_loadu_ps(p_in + 24), s3);
    }
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));

    for (; rem >= 8; rem -= 8, p_in += 8) [[unlikely]]
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in), _mm256_loadu_ps(p_in), s0);

    __m128 v_low = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    v_low = _mm_hadd_ps(v_low, v_low);
    v_low = _mm_hadd_ps(v_low, v_low);
    float sq_sum = _mm_cvtss_f32(v_low);
    for (size_t i = 0; i < rem; ++i) sq_sum += p_in[i] * p_in[i];

    // Hardware RSQRT in AVX2 
    float sum_val = sq_sum / size + eps;
    __m256 v_x = _mm256_set1_ps(sum_val);
    __m256 v_y0 = _mm256_rsqrt_ps(v_x);
    __m256 v_half_x = _mm256_mul_ps(v_x, _mm256_set1_ps(0.5f));
    __m256 v_inv = _mm256_mul_ps(v_y0, _mm256_fnmadd_ps(v_half_x, _mm256_mul_ps(v_y0, v_y0), _mm256_set1_ps(1.5f)));
    
    p_in = in; rem = size;
    for (; rem >= 32; rem -= 32, p_in += 32, p_w += 32, p_out += 32) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_in + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_w + 64), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(p_out + 64), _MM_HINT_T0);

        _mm256_storeu_ps(p_out,      _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in),      v_inv), _mm256_loadu_ps(p_w)));
        _mm256_storeu_ps(p_out + 8,  _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in + 8),  v_inv), _mm256_loadu_ps(p_w + 8)));
        _mm256_storeu_ps(p_out + 16, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in + 16), v_inv), _mm256_loadu_ps(p_w + 16)));
        _mm256_storeu_ps(p_out + 24, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in + 24), v_inv), _mm256_loadu_ps(p_w + 24)));
    }
    
    for (; rem >= 8; rem -= 8, p_in += 8, p_w += 8, p_out += 8) [[unlikely]]
        _mm256_storeu_ps(p_out, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in), v_inv), _mm256_loadu_ps(p_w)));

    float inv_s = _mm256_cvtss_f32(v_inv);
    for (size_t i = 0; i < rem; ++i) p_out[i] = p_in[i] * inv_s * p_w[i];
}

void silu(float* __restrict data, size_t size) {
    float* p_data = data;
    size_t rem = size;

    for (; rem >= 32; rem -= 32, p_data += 32) [[likely]] {
        _mm_prefetch(reinterpret_cast<const char*>(p_data + 64), _MM_HINT_T0);
        
        _mm256_storeu_ps(p_data,      silu_ps(_mm256_loadu_ps(p_data)));
        _mm256_storeu_ps(p_data + 8,  silu_ps(_mm256_loadu_ps(p_data + 8)));
        _mm256_storeu_ps(p_data + 16, silu_ps(_mm256_loadu_ps(p_data + 16)));
        _mm256_storeu_ps(p_data + 24, silu_ps(_mm256_loadu_ps(p_data + 24)));
    }

    for (; rem >= 8; rem -= 8, p_data += 8) [[unlikely]]
        _mm256_storeu_ps(p_data, silu_ps(_mm256_loadu_ps(p_data)));

    // Fast scalar remainder
    for (size_t i = 0; i < rem; ++i) [[unlikely]]
        p_data[i] = p_data[i] / (1.0f + std::exp(-p_data[i]));
}

} // namespace nca::simd::avx2
