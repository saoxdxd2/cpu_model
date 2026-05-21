#pragma once
// ============================================================================
// NCA — Kronecker-Factored Recursive Least Squares (RLS)
// core/spectral/kronecker_rls.hpp
// ============================================================================

#include "core/simd/memory.hpp"
#include <vector>

namespace nca::spectral {

// ── KRONECKER RLS STATE (v8.1 - KFW Edition) ───────────────────────────────
// Represents a compressed weight matrix W = Wa ⊗ Wb.
// This allows the model to learn correlations between different frequencies.
struct KroneckerRLSState {
    size_t d_model;
    size_t dim_a; // 64
    size_t dim_b; // 32

    // Weight Factors (Wa ⊗ Wb)
    nca::simd::aligned_unique_ptr<float[]> Wa;
    nca::simd::aligned_unique_ptr<float[]> Wb;

    // Covariance Factors (Inverse Covariances for RLS)
    nca::simd::aligned_unique_ptr<float[]> A;
    nca::simd::aligned_unique_ptr<float[]> B;

    explicit KroneckerRLSState(size_t d);

    // Resets to deterministic identity state.
    void reset();

    // Performs the K-RLS update using the tensor trick.
    void update(const float* x, const float* target, float lambda, float eta);

    // Applies the Kronecker weight operator: y = vec(Wb * X_mat * Wa^T)
    void apply(const float* x, float* out) const;
};

} // namespace nca::spectral
