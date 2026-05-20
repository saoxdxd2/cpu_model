#pragma once
// ============================================================================
// NCA — AVX2 Inline Math Approximations
// core/simd/avx2_math.hpp
// ============================================================================

#include <immintrin.h>

namespace nca::simd::avx2 {

/// Branchless Exponential Approximation (Minimax + float bit manipulation)
inline __m256 exp_ps(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    
    __m256 y = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    __m256 e = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 z = _mm256_sub_ps(y, e);
    
    __m256 p = _mm256_set1_ps(0.0001540332786645f);
    p = _mm256_fmadd_ps(z, p, _mm256_set1_ps(0.0013338381028420f));
    p = _mm256_fmadd_ps(z, p, _mm256_set1_ps(0.0096181291076284f));
    p = _mm256_fmadd_ps(z, p, _mm256_set1_ps(0.0555041086648215f));
    p = _mm256_fmadd_ps(z, p, _mm256_set1_ps(0.2402265069591007f));
    p = _mm256_fmadd_ps(z, p, _mm256_set1_ps(0.6931471805599453f));
    p = _mm256_fmadd_ps(z, p, _mm256_set1_ps(1.0f));
    
    // Construct 2^e using bit manipulation
    __m256i ei = _mm256_cvtps_epi32(e);
    ei = _mm256_add_epi32(ei, _mm256_set1_epi32(127));
    ei = _mm256_slli_epi32(ei, 23);
    __m256 p2 = _mm256_castsi256_ps(ei);
    
    return _mm256_mul_ps(p, p2);
}

/// Branchless SiLU Activation
inline __m256 silu_ps(__m256 x) {
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    __m256 den = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_ps(neg_x));
    
    // Fast division
    __m256 y0 = _mm256_rcp_ps(den);
    __m256 y1 = _mm256_mul_ps(y0, _mm256_fnmadd_ps(den, y0, _mm256_set1_ps(2.0f)));
    
    return _mm256_mul_ps(x, y1);
}

} // namespace nca::simd::avx2
