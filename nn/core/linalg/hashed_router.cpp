// ============================================================================
// NCA — Hashed Logic Router Implementation
// core/linalg/hashed_router.cpp
// ============================================================================

#include "core/linalg/hashed_router.hpp"
#include "core/simd/avx512_kernels.hpp"
#include <random>
#include <algorithm>

namespace nca::linalg {

HashedRouter::HashedRouter(Config cfg) : cfg_(cfg) {
    projections_ = nca::simd::make_aligned_unique<float[]>(cfg.n_experts * cfg.d_model);
    
    // Initialize with "Neural DNA" (Randomized Anchors)
    std::mt19937 gen(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for(size_t i=0; i < cfg.n_experts * cfg.d_model; ++i) {
        projections_[i] = dist(gen);
    }
}

void HashedRouter::route(const float* x, std::vector<size_t>& out_indices) const {
    alignas(64) float scores[4096]; // Assume max 4096 experts for L2/L3 fit
    const size_t n = std::min(cfg_.n_experts, (size_t)4096);

    // ── SILICON SATURATION: ROUTER SCAN ─────────────────────────────────────
    // We use a simplified inner-product scan.
    // In a production version, this would be a Rank-16 VNNI hash.
    for (size_t i = 0; i < n; ++i) {
        float score = 0.0f;
        const float* p = &projections_[i * cfg_.d_model];
        // Use auto-vectorized loop
        for (size_t j = 0; j < cfg_.d_model; ++j) score += p[j] * x[j];
        scores[i] = score;
    }

    // ── COMPLEXITY DESTRUCTION: QUICK-SELECT ────────────────────────────────
    // Replace O(N log N) sort with O(N) selection.
    std::vector<std::pair<float, size_t>> ranked(n);
    for(size_t i=0; i<n; ++i) ranked[i] = {scores[i], i};

    std::nth_element(ranked.begin(), ranked.begin() + cfg_.top_k, ranked.end(),
        [](auto& a, auto& b) { return a.first > b.first; });

    out_indices.clear();
    for(size_t i=0; i < cfg_.top_k; ++i) {
        out_indices.push_back(ranked[i].second);
    }
}

} // namespace nca::linalg
