#pragma once
// ============================================================================
// NCA — Latent Adapter
// core/execution/latent_adapter.hpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include "config/model_config.hpp"
#include <vector>

namespace nca::execution {

class LatentAdapter {
public:
    explicit LatentAdapter(size_t in_dim = nca::config::D_MODEL);

    // Performs the Saturated Rank-16 Projection: Vision -> Logic
    void project(const float* in, float* out);

private:
    size_t in_dim_;
    // Using vector to handle destructors of MXINT8Tensor correctly
    std::vector<nca::linalg::MXINT8Tensor> weights_;
};

} // namespace nca::execution
