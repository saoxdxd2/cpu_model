#pragma once
// ============================================================================
// NCA — Latent Adapter
// core/execution/latent_adapter.hpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include "core/simd/memory.hpp"
#include <memory>

namespace nca::execution {

class LatentAdapter {
public:
    LatentAdapter();

    // Performs the Saturated Rank-16 Projection: Vision -> Logic
    void project(const float* in, float* out);

private:
    // Aligned weights for the adapter core
    nca::simd::aligned_unique_ptr<nca::linalg::MXINT8Tensor[]> weights_;
};

} // namespace nca::execution
