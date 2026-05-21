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
    // Allocate raw memory for array of MXINT8Tensor
    void* ptr = _aligned_malloc(nca::config::D_MODEL * sizeof(nca::linalg::MXINT8Tensor), 64);
    if (!ptr) throw std::bad_alloc();
    nca::linalg::MXINT8Tensor* tensors = static_cast<nca::linalg::MXINT8Tensor*>(ptr);
    
    // Explicitly construct each element
    for(size_t i=0; i<nca::config::D_MODEL; ++i) {
        new (&tensors[i]) nca::linalg::MXINT8Tensor(nca::config::D_MODEL);
    }
    weights_.reset(tensors);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.02f, 0.02f);
    
    alignas(64) float temp[2048]; 
    for(size_t i=0; i<nca::config::D_MODEL; ++i) {
        for(size_t j=0; j<nca::config::D_MODEL; ++j) temp[j] = dist(gen);
        nca::linalg::mx_quantize_w(temp, weights_[i]);
    }
}

void LatentAdapter::project(const float* in, float* out) {
    nca::linalg::MXUINT8Tensor in_q(nca::config::D_MODEL);
    nca::linalg::mx_quantize_x(in, in_q);

    for (size_t i = 0; i < nca::config::D_MODEL; i += 16) {
        nca::linalg::mx_rank16_dot(&weights_[i], in_q, &out[i]);
    }
}

} // namespace nca::execution
