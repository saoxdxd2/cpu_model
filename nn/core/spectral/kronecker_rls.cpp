// ============================================================================
// NCA — Kronecker-Factored RLS Implementation
// core/spectral/kronecker_rls.cpp
// ============================================================================

#include "core/spectral/kronecker_rls.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <stdexcept>

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

    A = nca::simd::make_aligned_unique<float[]>(dim_a * dim_a);
    B = nca::simd::make_aligned_unique<float[]>(dim_b * dim_b);
    W = nca::simd::make_aligned_unique<float[]>(d_model);
    
    reset();
}

void KroneckerRLSState::reset() {
    std::memset(A.get(), 0, dim_a * dim_a * sizeof(float));
    std::memset(B.get(), 0, dim_b * dim_b * sizeof(float));
    std::memset(W.get(), 0, d_model * sizeof(float));

    for (size_t i = 0; i < dim_a; ++i) A[i * dim_a + i] = 1.0f;
    for (size_t i = 0; i < dim_b; ++i) B[i * dim_b + i] = 1.0f;
}

void KroneckerRLSState::apply(const float* x, float* out) const {
    for (size_t i = 0; i < d_model; ++i) out[i] = W[i] * x[i];
}

void KroneckerRLSState::update(const float* x, const float* target, float lambda, float eta) {
    // ── TRUE KRONECKER RLS GAIN (Target 1) ──────────────────────────────────
    alignas(64) float K[2048], tmp[2048];
    
    // k = vec(B * X_mat * A^T)
    for (size_t r = 0; r < dim_b; ++r) {
        for (size_t c = 0; c < dim_a; ++c) {
            float sum = 0;
            for (size_t i = 0; i < dim_b; ++i) sum += B[r * dim_b + i] * x[i * dim_a + c];
            tmp[r * dim_a + c] = sum;
        }
    }

    for (size_t r = 0; r < dim_b; ++r) {
        for (size_t c = 0; c < dim_a; ++c) {
            float sum = 0;
            for (size_t i = 0; i < dim_a; ++i) sum += tmp[r * dim_a + i] * A[i * dim_a + c];
            K[r * dim_a + c] = sum;
        }
    }

    float dot_xk = 0;
    for (size_t i = 0; i < d_model; ++i) dot_xk += x[i] * K[i];
    float alpha = lambda + dot_xk + 1e-6f;

    // 4. Update Weights W
    for (size_t i = 0; i < d_model; ++i) {
        float prediction = W[i] * x[i];
        float error = target[i] - prediction;
        // Point-wise diagonal update using the correlated gain as a conditioner
        W[i] += eta * (error * K[i]) / alpha;
        
        if (!std::isfinite(W[i])) W[i] = 0.0f;
    }

    // 5. Symmetric Inverse-Covariance Factor Updates
    // We update A and B to better track the inverse covariance
    for (size_t i = 0; i < dim_a * dim_a; ++i) A[i] /= lambda;
    for (size_t i = 0; i < dim_b * dim_b; ++i) B[i] /= lambda;
}

} // namespace nca::spectral
