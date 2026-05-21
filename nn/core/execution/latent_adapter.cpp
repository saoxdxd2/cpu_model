// ============================================================================
// NCA — Latent Adapter
// core/execution/latent_adapter.cpp
// ============================================================================

#include "core/execution/latent_adapter.hpp"
#include "core/linalg/mx_linear.hpp"
#include "core/simd/memory.hpp"
#include "config/model_config.hpp"
#include <random>

namespace nca::execution {

LatentAdapter::LatentAdapter() {
    weights_.reserve(nca::config::D_MODEL);
    for(size_t i=0; i<nca::config::D_MODEL; ++i) {
        weights_.emplace_back(nca::config::D_MODEL / 32);
    }

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    
    auto temp = nca::simd::make_aligned_unique<float[]>(nca::config::D_MODEL);
    for(size_t i=0; i<nca::config::D_MODEL; ++i) {
        for(size_t j=0; j<nca::config::D_MODEL; ++j) temp[j] = dist(gen);
        nca::linalg::mx_quantize_w(temp.get(), weights_[i]);
    }
}

void LatentAdapter::project(const float* in, float* out) {
    nca::linalg::MXUINT8Tensor in_q(nca::config::D_MODEL / 32);
    nca::linalg::mx_quantize_x(in, in_q);

    for (size_t i = 0; i < nca::config::D_MODEL; i += 16) {
        nca::linalg::mx_rank16_dot(&weights_[i], in_q, &out[i]);
    }
}

} // namespace nca::execution
