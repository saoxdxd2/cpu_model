// ============================================================================
// NCA -- Halting Gate (Phase 7)
// core/layers/halting.cpp
//
// Adaptive Computation Time logic. 
// Uses VNNI dot product + branchless math for fast scalar classifier.
// ============================================================================

#include "core/layers/halting.hpp"
#include "core/simd/avx512_math.hpp"
#include <cmath>

namespace nca::layers {

float halting_step(
    const nca::linalg::MXUINT8Tensor& x_q,
    const nca::linalg::MXINT8Tensor& w_halt,
    float b_halt,
    HaltingState& state,
    bool& should_halt
) {
    float epsilon = 0.01f;
    float dot = nca::linalg::mx_dot(w_halt, x_q) + b_halt;
    
    // sigmoid
    float p_t = 1.0f / (1.0f + std::exp(-dot));
    
    if (state.p_sum + p_t >= 1.0f - epsilon) {
        p_t = 1.0f - state.p_sum;
        state.remainder = p_t;
        state.p_sum = 1.0f;
        should_halt = true;
    } else {
        state.p_sum += p_t;
        should_halt = false;
    }
    
    state.steps += 1;
    return p_t;
}

} // namespace nca::layers
