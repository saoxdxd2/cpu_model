#pragma once
// ============================================================================
// NCA — Kronecker-Factored Recursive Least Squares (RLS)
// core/spectral/kronecker_rls.hpp
// ============================================================================

#include "core/simd/memory.hpp"
#include <vector>

namespace nca::spectral {

// ── KRONECKER RLS STATE ─────────────────────────────────────────────────────
// Represents a compressed covariance matrix P = A ⊗ B.
// This allows O(N^3) intelligence at O(N log N) speed.
struct KroneckerRLSState {
    size_t d_model;
    size_t dim_a; // ~sqrt(D)
    size_t dim_b; // ~sqrt(D)

    // Factors A and B (Inverse Covariances)
    nca::simd::aligned_unique_ptr<float[]> A;
    nca::simd::aligned_unique_ptr<float[]> B;

    // Spectral weights W (in frequency domain)
    nca::simd::aligned_unique_ptr<float[]> W;

    explicit KroneckerRLSState(size_t d);

    // Resets to deterministic identity/zero state.
    void reset();

    // Performs the RLS update using the Kronecker tensor trick.
    // x: spectral input
    // target: spectral target (GLR proposal)
    void update(const float* x, const float* target, float lambda, float eta);

    // Applies the current spectral operator to an input.
    void apply(const float* x, float* out) const;
};

} // namespace nca::spectral
