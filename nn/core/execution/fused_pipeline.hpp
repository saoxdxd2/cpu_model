#pragma once
// ============================================================================
// NCA -- Fused Multimodal Pipeline
// core/execution/fused_pipeline.hpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"

namespace nca::execution {

// ── HORIZONTAL NANO-CORE FUSION (The "Google Technique") ─────────────────────
// Fuses the Vision Core and Logic Core into a single pipeline.
void multimodal_fused_step(
    const float* __restrict text_in,
    const float* __restrict img_in,
    const nca::linalg::MXINT8Tensor* __restrict W_gate,
    const nca::linalg::MXINT8Tensor* __restrict W_up,
    float* __restrict out
);

} // namespace nca::execution
