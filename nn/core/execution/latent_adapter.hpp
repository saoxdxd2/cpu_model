#pragma once
// ============================================================================
// NCA — Cross-Core Latent Adapter
// core/execution/latent_adapter.hpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include <memory>

namespace nca::execution {

// ── LATENT ADAPTER (Phase 12) ──────────────────────────────────────────────
// Projects representations between different core latent spaces (e.g. Vision -> Logic).
// Uses the Rank-16 Saturated Math Core for zero-latency bridging.
class LatentAdapter {
public:
    struct Config {
        size_t d_src;
        size_t d_dst;
    };

    explicit LatentAdapter(Config cfg);

    // Projects x_src into y_dst branchlessly.
    void project(
        const nca::linalg::MXUINT8Tensor& x_src,
        float* __restrict y_dst
    ) const;

    // Weight access for initialization
    nca::linalg::MXINT8Tensor& weights() { return weights_; }

private:
    Config cfg_;
    nca::linalg::MXINT8Tensor weights_;
};

} // namespace nca::execution
