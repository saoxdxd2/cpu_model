// ============================================================================
// NCA — Hashed Logic Router Implementation (v13.0 - Salience-Driven)
// core/linalg/hashed_router.cpp
// ============================================================================

#include "core/linalg/hashed_router.hpp"
#include "core/simd/avx512_vnni.hpp"
#include "core/linalg/mx_linear.hpp"
#include <random>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace nca::linalg {

HashedRouter::HashedRouter(Config cfg) : cfg_(cfg) {
    if (cfg.top_k > cfg.n_experts) {
        throw std::invalid_argument("top_k cannot exceed n_experts");
    }
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
    size_t k = cfg_.top_k;
    out_indices.resize(k);
    
    nca::linalg::MXUINT8Tensor x_q(cfg_.d_model / 32);
    nca::linalg::mx_quantize_x(x, x_q);
    
    size_t count = 0;
    route_to_buffer(x_q, out_indices.data(), &count);
    out_indices.resize(count);
}

void HashedRouter::route_to_buffer(const nca::linalg::MXUINT8Tensor& x_q, size_t* out_buffer, size_t* out_count) const {
    const size_t N = cfg_.n_experts;
    const size_t k_target = cfg_.top_k;

    // 2. Score Generation (Saturated VNNI)
    alignas(64) float scores[1024]; // Max 1024 experts
    const size_t scan_limit = std::min(N, (size_t)1024);

    size_t i = 0;
    for (; i + 15 < scan_limit; i += 16) {
        nca::simd::avx512::rank16_vnni_dot(&projections_[i], &x_q, &scores[i]);
    }
    for (; i < scan_limit; ++i) {
        scores[i] = nca::linalg::mx_dot(projections_[i], x_q);
    }

    // 3. Selection (O(NK) via simple linear scan to avoid allocations)
    *out_count = std::min(k_target, scan_limit);
    for (size_t j = 0; j < *out_count; ++j) {
        float best_s = -1e9f;
        size_t best_idx = 0;
        for (size_t m = 0; m < scan_limit; ++m) {
            if (scores[m] > best_s) {
                best_s = scores[m];
                best_idx = m;
            }
        }
        out_buffer[j] = best_idx;
        scores[best_idx] = -1e9f; // Mark as selected
    }
}

} // namespace nca::linalg
