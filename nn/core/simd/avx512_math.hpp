#pragma once
// ============================================================================
// NCA — AVX-512 Inline Math Approximations
// core/simd/avx512_math.hpp
//
// Extremely optimized, branchless mathematical primitives for __m512 vectors.
// Exposing these directly in the header allows them to be aggressively 
// inlined into any future specialized loop without function call overhead.
// ============================================================================

#include <immintrin.h>

namespace nca::simd::avx512 {

/// Branchless Exponential Approximation (Minimax + scalef)
inline __m512 exp_ps(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-88.3762626647949f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.3762626647949f));
    
    __m512 y = _mm512_mul_ps(x, _mm512_set1_ps(1.4426950408889634f));
    __m512 e = _mm512_roundscale_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 z = _mm512_sub_ps(y, e);
    
    __m512 p = _mm512_set1_ps(0.0001540332786645f);
    p = _mm512_fmadd_ps(z, p, _mm512_set1_ps(0.0013338381028420f));
    p = _mm512_fmadd_ps(z, p, _mm512_set1_ps(0.0096181291076284f));
    p = _mm512_fmadd_ps(z, p, _mm512_set1_ps(0.0555041086648215f));
    p = _mm512_fmadd_ps(z, p, _mm512_set1_ps(0.2402265069591007f));
    p = _mm512_fmadd_ps(z, p, _mm512_set1_ps(0.6931471805599453f));
    p = _mm512_fmadd_ps(z, p, _mm512_set1_ps(1.0f));
    
    return _mm512_scalef_ps(p, e);
}

/// Branchless SiLU Activation
inline __m512 silu_ps(__m512 x) {
    __m512 neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    __m512 den = _mm512_add_ps(_mm512_set1_ps(1.0f), exp_ps(neg_x));
    
    __m512 y0 = _mm512_rcp14_ps(den);
    __m512 y1 = _mm512_mul_ps(y0, _mm512_fnmadd_ps(den, y0, _mm512_set1_ps(2.0f)));
    
    return _mm512_mul_ps(x, y1);
}

} // namespace nca::simd::avx512
