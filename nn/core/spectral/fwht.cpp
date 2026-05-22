// ============================================================================
// NCA — Fast Walsh-Hadamard Transform (FWHT) Implementation
// core/spectral/fwht.cpp
// ============================================================================

#include "core/spectral/fwht.hpp"
#include <immintrin.h>
#include <stdexcept>

namespace nca::spectral {

#if defined(__AVX512F__) || defined(_MSC_VER)
inline void butterfly_v16(float* __restrict a, float* __restrict b) {
    __m512 vA = _mm512_loadu_ps(a);
    __m512 vB = _mm512_loadu_ps(b);
    _mm512_storeu_ps(a, _mm512_add_ps(vA, vB));
    _mm512_storeu_ps(b, _mm512_sub_ps(vA, vB));
}
#endif

inline void butterfly_v8(float* __restrict a, float* __restrict b) {
    __m256 vA = _mm256_loadu_ps(a);
    __m256 vB = _mm256_loadu_ps(b);
    _mm256_storeu_ps(a, _mm256_add_ps(vA, vB));
    _mm256_storeu_ps(b, _mm256_sub_ps(vA, vB));
}

void fwht_inplace(std::span<float> data) {
    const size_t n = data.size();
    if ((n & (n - 1)) != 0) throw std::invalid_argument("FWHT size must be power of 2");

    float* p = data.data();

    // ── AVX2 BASELINE: Branchless and Scaled ────────────────────────────────
    for (size_t len = 1; len < n; len <<= 1) {
        for (size_t i = 0; i < n; i += (len << 1)) {
#if defined(__AVX512F__) || defined(_MSC_VER)
            if (len >= 16) {
                for (size_t j = 0; j < len; j += 16) {
                    butterfly_v16(p + i + j, p + i + len + j);
                }
                continue;
            }
#endif
            if (len >= 8) {
                for (size_t j = 0; j < len; j += 8) {
                    butterfly_v8(p + i + j, p + i + len + j);
                }
            } else {
                for (size_t j = 0; j < len; ++j) {
                    float a = p[i + j];
                    float b = p[i + len + j];
                    p[i + j] = a + b;
                    p[i + len + j] = a - b;
                }
            }
        }
    }
}

void ifwht_inplace(std::span<float> data) {
    const size_t n = data.size();
    fwht_inplace(data);
    
    float inv_n = 1.0f / (float)n;
#if defined(__AVX512F__) || defined(_MSC_VER)
    __m512 v_inv = _mm512_set1_ps(inv_n);
    for (size_t i = 0; i < n; i += 16) {
        _mm512_storeu_ps(data.data() + i, _mm512_mul_ps(_mm512_loadu_ps(data.data() + i), v_inv));
    }
#else
    __m256 v_inv = _mm256_set1_ps(inv_n);
    for (size_t i = 0; i < n; i += 8) {
        _mm256_storeu_ps(data.data() + i, _mm256_mul_ps(_mm256_loadu_ps(data.data() + i), v_inv));
    }
#endif
}

void ifwht_no_scale(std::span<float> data) {
    fwht_inplace(data);
}

} // namespace nca::spectral

