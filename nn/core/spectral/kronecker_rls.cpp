// ============================================================================
// NCA — Kronecker-Factored RLS Implementation (v8.1 - KFW)
// core/spectral/kronecker_rls.cpp
// ============================================================================

#include "core/spectral/kronecker_rls.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <immintrin.h>
#include <bit>

namespace nca::spectral {

KroneckerRLSState::KroneckerRLSState(size_t d) : d_model(d) {
    if (d == 2048) { dim_a = 64; dim_b = 32; }
    else if (d == 16) { dim_a = 4; dim_b = 4; } 
    else {
        dim_a = (size_t)std::sqrt(d);
        while (d % dim_a != 0) dim_a--;
        dim_b = d / dim_a;
    }
    if (dim_a * dim_b != d_model) throw std::invalid_argument("d_model must equal dim_a * dim_b");

    Wa = nca::simd::make_aligned_unique<float[]>(dim_a * dim_a);
    Wb = nca::simd::make_aligned_unique<float[]>(dim_b * dim_b);
    A = nca::simd::make_aligned_unique<float[]>(dim_a * dim_a);
    B = nca::simd::make_aligned_unique<float[]>(dim_b * dim_b);
    
    reset();
}

void KroneckerRLSState::reset() {
    std::memset(Wa.get(), 0, dim_a * dim_a * sizeof(float));
    std::memset(Wb.get(), 0, dim_b * dim_b * sizeof(float));
    std::memset(A.get(), 0, dim_a * dim_a * sizeof(float));
    std::memset(B.get(), 0, dim_b * dim_b * sizeof(float));

    // Initialize Weights and Covariances to Identity
    for (size_t i = 0; i < dim_a; ++i) { Wa[i * dim_a + i] = 1.0f; A[i * dim_a + i] = 1.0f; }
    for (size_t i = 0; i < dim_b; ++i) { Wb[i * dim_b + i] = 1.0f; B[i * dim_b + i] = 1.0f; }
}

void KroneckerRLSState::apply(const float* x, float* out) const {
    alignas(64) float tmp[2048];
    std::memset(tmp, 0, d_model * sizeof(float));
    
    // 1. tmp = Wb * X_mat (Broadcast Wb, Vectorize X)
    for (size_t r = 0; r < dim_b; ++r) {
        for (size_t i = 0; i < dim_b; ++i) {
            __m256 wb_val = _mm256_set1_ps(Wb[r * dim_b + i]);
            size_t c = 0;
            for (; c + 7 < dim_a; c += 8) {
                __m256 x_val = _mm256_loadu_ps(&x[i * dim_a + c]);
                __m256 t_val = _mm256_loadu_ps(&tmp[r * dim_a + c]);
                _mm256_storeu_ps(&tmp[r * dim_a + c], _mm256_fmadd_ps(wb_val, x_val, t_val));
            }
            for (; c < dim_a; ++c) tmp[r * dim_a + c] += Wb[r * dim_b + i] * x[i * dim_a + c];
        }
    }

    // 2. out = tmp * Wa^T (Contiguous dot product)
    for (size_t r = 0; r < dim_b; ++r) {
        for (size_t c = 0; c < dim_a; ++c) {
            __m256 sum256 = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 7 < dim_a; i += 8) {
                __m256 tmp_val = _mm256_loadu_ps(&tmp[r * dim_a + i]);
                __m256 wa_val = _mm256_loadu_ps(&Wa[c * dim_a + i]);
                sum256 = _mm256_fmadd_ps(tmp_val, wa_val, sum256);
            }
            __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float s = _mm_cvtss_f32(sum128);
            for (; i < dim_a; ++i) s += tmp[r * dim_a + i] * Wa[c * dim_a + i];
            out[r * dim_a + c] = s;
        }
    }
}

void KroneckerRLSState::update(const float* x, const float* target, float lambda, float eta) {
    alignas(64) float K[2048], tmp[2048], prediction[2048], e[2048];
    std::memset(tmp, 0, d_model * sizeof(float));
    
    // 1. Compute Gain K = (A ⊗ B) * x
    // tmp = B * x
    for (size_t r = 0; r < dim_b; ++r) {
        for (size_t i = 0; i < dim_b; ++i) {
            __m256 b_val = _mm256_set1_ps(B[r * dim_b + i]);
            size_t c = 0;
            for (; c + 7 < dim_a; c += 8) {
                __m256 x_val = _mm256_loadu_ps(&x[i * dim_a + c]);
                __m256 t_val = _mm256_loadu_ps(&tmp[r * dim_a + c]);
                _mm256_storeu_ps(&tmp[r * dim_a + c], _mm256_fmadd_ps(b_val, x_val, t_val));
            }
            for (; c < dim_a; ++c) tmp[r * dim_a + c] += B[r * dim_b + i] * x[i * dim_a + c];
        }
    }
    
    // K = tmp * A^T (contiguous dot product)
    for (size_t r = 0; r < dim_b; ++r) {
        for (size_t c = 0; c < dim_a; ++c) {
            __m256 sum256 = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 7 < dim_a; i += 8) {
                __m256 tmp_val = _mm256_loadu_ps(&tmp[r * dim_a + i]);
                __m256 a_val = _mm256_loadu_ps(&A[c * dim_a + i]);
                sum256 = _mm256_fmadd_ps(tmp_val, a_val, sum256);
            }
            __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float s = _mm_cvtss_f32(sum128);
            for (; i < dim_a; ++i) s += tmp[r * dim_a + i] * A[c * dim_a + i];
            K[r * dim_a + c] = s;
        }
    }

    __m256 v_dot_sum256 = _mm256_setzero_ps();
    size_t d = 0;
    for (; d + 7 < d_model; d += 8) {
        v_dot_sum256 = _mm256_fmadd_ps(_mm256_loadu_ps(&x[d]), _mm256_loadu_ps(&K[d]), v_dot_sum256);
    }
    __m128 v_dot_sum128 = _mm_add_ps(_mm256_castps256_ps128(v_dot_sum256), _mm256_extractf128_ps(v_dot_sum256, 1));
    v_dot_sum128 = _mm_hadd_ps(v_dot_sum128, v_dot_sum128);
    v_dot_sum128 = _mm_hadd_ps(v_dot_sum128, v_dot_sum128);
    float dot_xk = _mm_cvtss_f32(v_dot_sum128);
    for (; d < d_model; ++d) dot_xk += x[d] * K[d];
    
    float alpha = lambda + dot_xk + 1e-4f;

    // 2. Compute Prediction and Error
    apply(x, prediction);
    for (d = 0; d + 7 < d_model; d += 8) {
        _mm256_storeu_ps(&e[d], _mm256_sub_ps(_mm256_loadu_ps(&target[d]), _mm256_loadu_ps(&prediction[d])));
    }
    for (; d < d_model; ++d) e[d] = target[d] - prediction[d];

    // 3. Update Weight Factors Wa and Wb
    float inv_alpha = eta / alpha;
    
    for (size_t r = 0; r < dim_a; ++r) {
        alignas(64) float dWa[2048]; 
        std::memset(dWa, 0, dim_a * sizeof(float));
        for (size_t i = 0; i < dim_b; ++i) {
            __m256 e_val = _mm256_set1_ps(e[i * dim_a + r]);
            size_t c = 0;
            for (; c + 7 < dim_a; c += 8) {
                __m256 k_val = _mm256_loadu_ps(&K[i * dim_a + c]);
                __m256 dwa_val = _mm256_loadu_ps(&dWa[c]);
                _mm256_storeu_ps(&dWa[c], _mm256_fmadd_ps(e_val, k_val, dwa_val));
            }
            for (; c < dim_a; ++c) dWa[c] += e[i * dim_a + r] * K[i * dim_a + c];
        }
        for (size_t c = 0; c < dim_a; ++c) {
            Wa[r * dim_a + c] += dWa[c] * inv_alpha;
            uint32_t val_bits = std::bit_cast<uint32_t>(Wa[r * dim_a + c]);
            uint32_t is_finite_mask = ((val_bits & 0x7F800000) != 0x7F800000) ? 0xFFFFFFFF : 0;
            uint32_t default_val = (r == c) ? 0x3F800000 : 0; 
            Wa[r * dim_a + c] = std::bit_cast<float>((val_bits & is_finite_mask) | (default_val & ~is_finite_mask));
        }
    }
    
    for (size_t r = 0; r < dim_b; ++r) {
        for (size_t c = 0; c < dim_b; ++c) {
            __m256 sum256 = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 7 < dim_a; i += 8) {
                __m256 e_val = _mm256_loadu_ps(&e[r * dim_a + i]);
                __m256 k_val = _mm256_loadu_ps(&K[c * dim_a + i]);
                sum256 = _mm256_fmadd_ps(e_val, k_val, sum256);
            }
            __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(sum256), _mm256_extractf128_ps(sum256, 1));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float s = _mm_cvtss_f32(sum128);
            for (; i < dim_a; ++i) s += e[r * dim_a + i] * K[c * dim_a + i];
            
            Wb[r * dim_b + c] += s * inv_alpha;
            uint32_t val_bits = std::bit_cast<uint32_t>(Wb[r * dim_b + c]);
            uint32_t is_finite_mask = ((val_bits & 0x7F800000) != 0x7F800000) ? 0xFFFFFFFF : 0;
            uint32_t default_val = (r == c) ? 0x3F800000 : 0; 
            Wb[r * dim_b + c] = std::bit_cast<float>((val_bits & is_finite_mask) | (default_val & ~is_finite_mask));
        }
    }

    // 4. Woodbury Factor Decay & Explosion Prevention (Vectorized)
    float max_A = 0.0f, max_B = 0.0f;
    __m256 v_inv_lambda = _mm256_set1_ps(1.0f / lambda);
    
    size_t i = 0;
    for (; i + 7 < dim_a * dim_a; i += 8) {
        _mm256_storeu_ps(&A[i], _mm256_mul_ps(_mm256_loadu_ps(&A[i]), v_inv_lambda));
    }
    for (; i < dim_a * dim_a; ++i) A[i] /= lambda;
    for (i = 0; i < dim_a; ++i) max_A = std::max(max_A, A[i * dim_a + i]);

    i = 0;
    for (; i + 7 < dim_b * dim_b; i += 8) {
        _mm256_storeu_ps(&B[i], _mm256_mul_ps(_mm256_loadu_ps(&B[i]), v_inv_lambda));
    }
    for (; i < dim_b * dim_b; ++i) B[i] /= lambda;
    for (i = 0; i < dim_b; ++i) max_B = std::max(max_B, B[i * dim_b + i]);

    if (max_A > 1e4f) {
        __m256 v_scale = _mm256_set1_ps(1e4f / max_A);
        for (i = 0; i + 7 < dim_a * dim_a; i += 8) _mm256_storeu_ps(&A[i], _mm256_mul_ps(_mm256_loadu_ps(&A[i]), v_scale));
        for (; i < dim_a * dim_a; ++i) A[i] *= (1e4f / max_A);
    }
    if (max_B > 1e4f) {
        __m256 v_scale = _mm256_set1_ps(1e4f / max_B);
        for (i = 0; i + 7 < dim_b * dim_b; i += 8) _mm256_storeu_ps(&B[i], _mm256_mul_ps(_mm256_loadu_ps(&B[i]), v_scale));
        for (; i < dim_b * dim_b; ++i) B[i] *= (1e4f / max_B);
    }
}

} // namespace nca::spectral
