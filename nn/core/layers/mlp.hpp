#pragma once
// ============================================================================
// NCA -- Gated MLP Layer (Phase 7)
// core/layers/mlp.hpp
//
// Implements the SwiGLU / Gated MLP block:
//   gate = W_gate * x
//   up   = W_up * x
//   hidden = silu(gate) * up
//   y = W_down * hidden
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include <cstddef>

namespace nca::layers {

// Fused Gate + SiLU + Up + Quantize step:
// Computes: out_q = quantize( silu(gate) * up )
// This eliminates the round-trip to memory for the intermediate 'hidden' vector.
void fused_gated_silu_quantize(
    const float* __restrict gate,
    const float* __restrict up,
    nca::linalg::MXUINT8Tensor& out_q,
    size_t d_inner
);

// Gated MLP block using MXINT8 weights.
// x: [d_model]
// W_gate: [d_inner, d_model]
// W_up: [d_inner, d_model]
// W_down: [d_model, d_inner]
void gated_mlp_step(
    const float* __restrict x,
    float* __restrict gate_buf,
    float* __restrict up_buf,
    float* __restrict y,
    const nca::linalg::MXINT8Tensor& W_gate,
    const nca::linalg::MXINT8Tensor& W_up,
    const nca::linalg::MXINT8Tensor& W_down,
    nca::linalg::MXUINT8Tensor& x_q,
    nca::linalg::MXUINT8Tensor& hidden_q,
    size_t d_model,
    size_t d_inner
);

} // namespace nca::layers
