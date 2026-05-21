// ============================================================================
// NCA — Cross-Core Latent Adapter
// core/execution/latent_adapter.cpp
// ============================================================================

#include "core/execution/latent_adapter.hpp"
#include "core/simd/dispatch.hpp"

namespace nca::execution {

LatentAdapter::LatentAdapter(Config cfg) 
    : cfg_(cfg), weights_(cfg.d_dst * cfg.d_src / 32) {
    // Note: num_blocks = (rows * cols) / 32
    // Here we assume rows=d_dst, cols=d_src. 
    // weights_ was initialized with enough blocks for the whole matrix.
}

void LatentAdapter::project(
    const nca::linalg::MXUINT8Tensor& x_src,
    float* __restrict y_dst
) const {
    // ── REGISTER-SATURATED PROJECTION ────────────────────────────────────────
    // We use the Rank-16 Saturated GEMV to ensure the bridge between agents
    // is as fast as the agents themselves.
    nca::linalg::mx_gemv(weights_, x_src, y_dst, cfg_.d_dst, cfg_.d_src);
}

} // namespace nca::execution
