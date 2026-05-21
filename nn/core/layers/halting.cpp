// ============================================================================
// NCA -- Halting Gate (Phase 7)
// core/layers/halting.cpp
//
// Adaptive Computation Time logic.
// Uses VNNI dot product + fully branchless math for the halt decision.
// The if/else is replaced with conditional arithmetic using masks.
// ============================================================================

#include "core/layers/halting.hpp"
#include "core/simd/avx512_math.hpp"
#include <cmath>
#include <algorithm>

namespace nca::layers {

float halting_step(
    const nca::linalg::MXUINT8Tensor& x_q,
    const nca::linalg::MXINT8Tensor& w_halt,
    float b_halt,
    HaltingState& state,
    bool& should_halt
) {
    constexpr float epsilon = 0.01f;
    float dot = nca::linalg::mx_dot(w_halt, x_q) + b_halt;

    // Branchless sigmoid: 1 / (1 + exp(-dot))
    float p_t = 1.0f / (1.0f + std::exp(-dot));

    // Branchless halt decision:
    // If p_sum + p_t >= 1 - epsilon, we halt and clamp p_t.
    // Instead of if/else, compute both paths and select via mask.
    float threshold = 1.0f - epsilon;
    float p_clamped = 1.0f - state.p_sum;  // What p_t would be if we halt
    float exceeds = static_cast<float>(state.p_sum + p_t >= threshold); // 1.0 or 0.0

    // Branchless blend: result = exceeds * clamped + (1 - exceeds) * p_t
    p_t = exceeds * p_clamped + (1.0f - exceeds) * p_t;

    state.p_sum += p_t;
    state.remainder = exceeds * p_t;  // Only set remainder when halting
    should_halt = (exceeds > 0.5f);
    state.steps += 1;

    return p_t;
}

} // namespace nca::layers
