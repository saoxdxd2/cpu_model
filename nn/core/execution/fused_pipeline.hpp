#pragma once
// ============================================================================
// NCA -- Fused Multimodal Pipeline
// core/execution/fused_pipeline.hpp
// ============================================================================

#include "core/vision/scanner.hpp"
#include "core/layers/mlp.hpp"
#include "core/linalg/mx_linear.hpp"

namespace nca::execution {

// ── HORIZONTAL NANO-CORE FUSION (The "Google Technique") ─────────────────────
// Fuses the Vision Core (Scanner) and Logic Core (MLP) into a single pipeline.
// Intermediate activations are kept in L1/L2 and immediately quantized,
// eliminating redundant round-trips to RAM.
void multimodal_fused_step(
    const float* __restrict img_in,
    float* __restrict output,
    nca::vision::ScannerConfig v_cfg,
    const nca::linalg::MXINT8Tensor& W_gate,
    const nca::linalg::MXINT8Tensor& W_up,
    const nca::linalg::MXINT8Tensor& W_down,
    float* __restrict h_ssm,
    const float* __restrict A_ssm,
    const float* __restrict B_ssm,
    const float* __restrict C_ssm
);

} // namespace nca::execution
