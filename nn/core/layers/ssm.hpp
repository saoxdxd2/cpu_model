#pragma once
// ============================================================================
// NCA — Selective State-Space Model (SSM)
// core/layers/ssm.hpp
// ============================================================================

#include <cstddef>
#include "core/linalg/mx_linear.hpp"

namespace nca::layers {

// Performs a single temporal step of a Selective SSM (Mamba-style)
// h: Hidden state of shape [d_inner, d_state] (Continuous dynamics memory)
// A: State transition matrix (diagonal), shape [d_inner, d_state]
// B: Input projection, shape [d_state]
// C: Output projection, shape [d_state]
// x: Input vector, shape [d_inner]
// y: Output vector, shape [d_inner]
struct SSMConfig {
    size_t d_inner = 8192;
    size_t d_state = 16;
};

void ssm_step(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C,
    const float* __restrict x,
    float* __restrict y,
    SSMConfig cfg
);

// Fused SSM + SiLU + Quantize:
void mx_fused_ssm_silu_quantize_step(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C,
    const float* __restrict x,
    nca::linalg::MXUINT8Tensor& y_q,
    SSMConfig cfg
);

} // namespace nca::layers
