#pragma once
// ============================================================================
// NCA — Hashed Logic Router (Complexity Destruction)
// core/linalg/hashed_router.hpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include <vector>
#include <cstdint>

namespace nca::linalg {

// ── HASHED LOGIC ROUTER (Phase 15) ──────────────────────────────────────────
// Uses Randomized Projections (LSH) to select relevant logic blocks.
// This implements the "CPU Advantage": JUMPING instead of WALKING.
class HashedRouter {
public:
    struct Config {
        size_t d_model;
        size_t n_experts;
        size_t top_k;
        size_t n_hashes; // Number of random projections
    };

    explicit HashedRouter(Config cfg);

    // Projects x into hash space and returns the indices of the top-K experts.
    void route(const float* x, std::vector<size_t>& out_indices) const;

private:
    Config cfg_;
    // Random projection matrix (Hashed Anchors)
    nca::simd::aligned_unique_ptr<float[]> projections_;
};

} // namespace nca::linalg
