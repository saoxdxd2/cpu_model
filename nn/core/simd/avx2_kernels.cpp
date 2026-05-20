// ============================================================================
// NCA — AVX2 Math Kernels
// core/simd/avx2_kernels.cpp
// 
// Compiled with /arch:AVX2. Condensed 4x unrolled loops, pointer logic.
// ============================================================================

#include "core/simd/avx2_kernels.hpp"
#include <immintrin.h>
#include <cmath>

namespace nca::simd::avx2 {

void rmsnorm(float* out, const float* in, const float* weight, int size, float eps) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps(), s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    const float *p_in = in, *p_w = weight;
    float *p_out = out;
    int rem = size;

    // 4x unrolled FMA block
    for (; rem >= 32; rem -= 32, p_in += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in),      _mm256_loadu_ps(p_in),      s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in + 8),  _mm256_loadu_ps(p_in + 8),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in + 16), _mm256_loadu_ps(p_in + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in + 24), _mm256_loadu_ps(p_in + 24), s3);
    }
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));

    for (; rem >= 8; rem -= 8, p_in += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(p_in), _mm256_loadu_ps(p_in), s0);

    // Fast horizontal sum
    __m128 v_low = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1));
    v_low = _mm_hadd_ps(v_low, v_low);
    v_low = _mm_hadd_ps(v_low, v_low);
    float sq_sum = _mm_cvtss_f32(v_low);
    
    // Scalar tail
    for (int i = 0; i < rem; ++i) sq_sum += p_in[i] * p_in[i];

    __m256 v_inv = _mm256_set1_ps(1.0f / std::sqrt(sq_sum / size + eps));
    
    p_in = in; rem = size;
    for (; rem >= 32; rem -= 32, p_in += 32, p_w += 32, p_out += 32) {
        _mm256_storeu_ps(p_out,      _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in),      v_inv), _mm256_loadu_ps(p_w)));
        _mm256_storeu_ps(p_out + 8,  _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in + 8),  v_inv), _mm256_loadu_ps(p_w + 8)));
        _mm256_storeu_ps(p_out + 16, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in + 16), v_inv), _mm256_loadu_ps(p_w + 16)));
        _mm256_storeu_ps(p_out + 24, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in + 24), v_inv), _mm256_loadu_ps(p_w + 24)));
    }
    
    for (; rem >= 8; rem -= 8, p_in += 8, p_w += 8, p_out += 8)
        _mm256_storeu_ps(p_out, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(p_in), v_inv), _mm256_loadu_ps(p_w)));

    float inv_s = _mm256_cvtss_f32(v_inv);
    for (int i = 0; i < rem; ++i) p_out[i] = p_in[i] * inv_s * p_w[i];
}

} // namespace nca::simd::avx2
