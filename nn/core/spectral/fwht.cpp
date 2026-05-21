// ============================================================================
// NCA — Fast Walsh-Hadamard Transform (FWHT) Implementation
// core/spectral/fwht.cpp
// ============================================================================

#include "core/spectral/fwht.hpp"
#include <immintrin.h>
#include <stdexcept>
#include <cmath>

namespace nca::spectral {

// ── AVX-512 HADAMARD BUTTERFLY ──────────────────────────────────────────────
// Processes 16 butterflies in parallel.
// a = a + b, b = a - b
inline void butterfly_v16(float* __restrict a, float* __restrict b) {
    __m512 vA = _mm512_loadu_ps(a);
    __m512 vB = _mm512_loadu_ps(b);
    _mm512_storeu_ps(a, _mm512_add_ps(vA, vB));
    _mm512_storeu_ps(b, _mm512_sub_ps(vA, vB));
}

void fwht_inplace(std::span<float> data) {
    const size_t n = data.size();
    if ((n & (n - 1)) != 0) throw std::invalid_argument("FWHT size must be power of 2");

    float* p = data.data();

    // Standard iterative FWHT (Butterfly Network)
    for (size_t len = 1; len < n; len <<= 1) {
        for (size_t i = 0; i < n; i += (len << 1)) {
            // If length is at least 16, use AVX-512
            if (len >= 16) {
                for (size_t j = 0; j < len; j += 16) {
                    butterfly_v16(p + i + j, p + i + len + j);
                }
            } else {
                // Scalar fallback for small butterflies
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
    
    // Scale by 1/N
    float inv_n = 1.0f / (float)n;
    __m512 v_inv = _mm512_set1_ps(inv_n);
    for (size_t i = 0; i < n; i += 16) {
        _mm512_storeu_ps(data.data() + i, _mm512_mul_ps(_mm512_loadu_ps(data.data() + i), v_inv));
    }
}

void ifwht_no_scale(std::span<float> data) {
    fwht_inplace(data);
}

} // namespace nca::spectral
