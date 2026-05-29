#pragma once
// ============================================================================
// NCA -- Halting Gate (Phase 7)
// core/layers/halting.hpp
//
// Implements Adaptive Computation Time (ACT) Halting Gate.
// Computes halting probability via an entropy classifier to determine
// dynamic early-exit points, enabling compute to shift from heavy to light blocks.
// ============================================================================

//#include "core/linalg/mx_linear.hpp"
#include <cstddef>

namespace nca::layers {

struct HaltingState {
    float p_sum = 0.0f;
    float remainder = 0.0f;
    size_t steps = 0;
};

// Computes the halting probability for the current layer/step.
// Returns the halting probability p_t.
// float halting_step(
//     const nca::linalg::MXUINT8Tensor& x_q,
//     const nca::linalg::MXINT8Tensor& w_halt,
//     float b_halt,
//     HaltingState& state,
//     bool& should_halt
// );

} // namespace nca::layers
