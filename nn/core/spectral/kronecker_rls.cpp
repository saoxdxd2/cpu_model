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
    // ── KFW APPLY: y = vec(Wb * X_mat * Wa^T) ───────────────────────────────
    alignas(64) float tmp[2048];
    
    // 1. tmp = Wb * X_mat
    for (size_t r = 0; r < dim_b; ++r) {
        for (size_t c = 0; c < dim_a; ++c) {
            float s = 0;
            for (size_t i = 0; i < dim_b; ++i) s += Wb[r * dim_b + i] * x[i * dim_a + c];
            tmp[r * dim_a + c] = s;
        }
    }

    // 2. out = tmp * Wa^T
    for (size_t r = 0; r < dim_b; ++r) {
        for (size_t c = 0; c < dim_a; ++c) {
            float s = 0;
            for (size_t i = 0; i < dim_a; ++i) s += tmp[r * dim_a + i] * Wa[c * dim_a + i];
            out[r * dim_a + c] = s;
        }
    }
}

void KroneckerRLSState::update(const float* x, const float* target, float lambda, float eta) {
    // ── KFW RLS UPDATE ──────────────────────────────────────────────────────
    alignas(64) float K[2048], tmp[2048], prediction[2048], e[2048];
    
    // 1. Compute Gain K = (A ⊗ B) * x
    for (size_t r=0; r<dim_b; ++r) for (size_t c=0; c<dim_a; ++c) {
        float s = 0; for (size_t i=0; i<dim_b; ++i) s += B[r*dim_b + i] * x[i*dim_a + c];
        tmp[r*dim_a + c] = s;
    }
    for (size_t r=0; r<dim_b; ++r) for (size_t c=0; c<dim_a; ++c) {
        float s = 0; for (size_t i=0; i<dim_a; ++i) s += tmp[r*dim_a + i] * A[c*dim_a + i];
        K[r*dim_a + c] = s;
    }

    float dot_xk = 0; for (size_t i=0; i<d_model; ++i) dot_xk += x[i] * K[i];
    float alpha = lambda + dot_xk + 1e-4f;

    // 2. Compute Prediction and Error
    apply(x, prediction);
    for (size_t i=0; i<d_model; ++i) e[i] = target[i] - prediction[i];

    // 3. Update Weight Factors Wa and Wb (Simplified Factor Update)
    // We treat the error-gain product as a rank-1 update to the factors.
    for (size_t r=0; r<dim_a; ++r) for (size_t c=0; c<dim_a; ++c) {
        Wa[r*dim_a + c] += eta * (e[r % d_model] * K[c % d_model]) / alpha;
        if(!std::isfinite(Wa[r*dim_a + c])) Wa[r*dim_a + c] = (r == c) ? 1.0f : 0.0f;
    }
    for (size_t r=0; r<dim_b; ++r) for (size_t c=0; c<dim_b; ++c) {
        Wb[r*dim_b + c] += eta * (e[r % d_model] * K[c % d_model]) / alpha;
        if(!std::isfinite(Wb[r*dim_b + c])) Wb[r*dim_b + c] = (r == c) ? 1.0f : 0.0f;
    }

    // 4. Woodbury Factor Decay
    for (size_t i = 0; i < dim_a * dim_a; ++i) A[i] /= lambda;
    for (size_t i = 0; i < dim_b * dim_b; ++i) B[i] /= lambda;
}

} // namespace nca::spectral
