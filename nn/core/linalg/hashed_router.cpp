// ============================================================================
// NCA — Hashed Logic Router Implementation (Saturated VNNI)
// core/linalg/hashed_router.cpp
// ============================================================================

#include "core/linalg/hashed_router.hpp"
#include "core/simd/avx512_vnni.hpp"
#include "core/linalg/mx_linear.hpp"
#include <random>
#include <algorithm>
#include <stdexcept>

namespace nca::linalg {

HashedRouter::HashedRouter(Config cfg) : cfg_(cfg) {
    if (cfg.top_k > cfg.n_experts) {
        throw std::invalid_argument("top_k cannot exceed n_experts");
    }

    // Projections are stored as MXINT8 for VNNI saturation
    projections_ = nca::simd::make_aligned_unique<MXINT8Tensor[]>(cfg.n_experts);
    for(size_t i=0; i < cfg.n_experts; ++i) {
        projections_[i] = MXINT8Tensor(cfg.d_model / 32);
    }
    
    std::mt19937 gen(1337); 
    std::normal_distribution<float> dist(0.0f, 0.02f);
    auto temp = nca::simd::make_aligned_unique<float[]>(cfg.d_model);

    for(size_t i=0; i < cfg.n_experts; ++i) {
        for(size_t j=0; j < cfg.d_model; ++j) temp[j] = dist(gen);
        nca::linalg::mx_quantize_w(temp.get(), projections_[i]);
    }
}

void HashedRouter::route(const float* x, std::vector<size_t>& out_indices) const {
    const size_t D = cfg_.d_model;
    const size_t N = cfg_.n_experts;

    MXUINT8Tensor x_q(D / 32);
    nca::linalg::mx_quantize_x(x, x_q);

    alignas(64) float scores[4096]; 
    const size_t scan_limit = std::min(N, (size_t)4096);

    // ── ROBUST RANK-16 SCAN ─────────────────────────────────────────────────
    size_t i = 0;
    for (; i + 15 < scan_limit; i += 16) {
        nca::simd::avx512::rank16_vnni_dot(&projections_[i], &x_q, &scores[i]);
    }
    // Handle remainders (Scalar fallback for edge cases)
    for (; i < scan_limit; ++i) {
        scores[i] = nca::linalg::mx_dot(projections_[i], x_q);
    }

    // ── COMPLEXITY DESTRUCTION: O(N) Top-K Selection ────────────────────────
    std::vector<std::pair<float, size_t>> ranked(scan_limit);
    for(size_t k=0; k<scan_limit; ++k) ranked[k] = {scores[k], k};

    const size_t k = std::min(cfg_.top_k, scan_limit);
    std::nth_element(ranked.begin(), ranked.begin() + k, ranked.end(),
        [](auto& a, auto& b) { return a.first > b.first; });

    out_indices.clear();
    for(size_t j=0; j < k; ++j) {
        out_indices.push_back(ranked[j].second);
    }
}

} // namespace nca::linalg
