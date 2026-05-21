#pragma once
// ============================================================================
// NCA — Latent Adapter
// core/execution/latent_adapter.hpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include <vector>

namespace nca::execution {

class LatentAdapter {
public:
    LatentAdapter();

    // Performs the Saturated Rank-16 Projection: Vision -> Logic
    void project(const float* in, float* out);

private:
    // Using vector to handle destructors of MXINT8Tensor correctly
    std::vector<nca::linalg::MXINT8Tensor> weights_;
};

} // namespace nca::execution
