#pragma once
// ============================================================================
// NCA — Selective State-Space Model (SSM)
// core/layers/ssm.hpp
// ============================================================================

#include <cstddef>
//#include "core/linalg/mx_linear.hpp"

namespace nca::layers {

struct SSMConfig {
    size_t d_inner;
    size_t d_state = 16;
};

// Performs a single selective step of the SSM.
// Equation:
//   h_t = A*h_{t-1} + B*x_t
//   y_t = C*h_t
void ssm_step(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C,
    const float* __restrict x,
    float* __restrict y,
    SSMConfig cfg
);

// Fused SSM + Quantization for multimodal pipeline efficiency.
// This is the "Horizontal Fusion" tweak to keep intermediate state in L1.
// void mx_fused_ssm_silu_quantize_step(
//     float* __restrict h,
//     const float* __restrict A,
//     const float* __restrict B,
//     const float* __restrict C,
//     const float* __restrict x,
//     nca::linalg::MXUINT8Tensor& y_q,
//     SSMConfig cfg
// );

} // namespace nca::layers
