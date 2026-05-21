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

    // ── BRANCHLESS SIGMOID TWEAK (Complexity Destruction) ────────────────────
    // Replaced scalar std::exp with the high-throughput AVX-512 minimax 
    // sigmoid approximation. This avoids the scalar FPU stall.
    alignas(64) float dot_buf[16] = {dot};
    auto v_dot = _mm512_load_ps(dot_buf);
    
    // Sigmoid(x) = 0.5 * tanh(0.5 * x) + 0.5
    // But we use a direct 1 / (1 + exp(-x)) approximation.
    auto v_sig = _mm512_rcp14_ps(_mm512_add_ps(_mm512_set1_ps(1.0f), nca::simd::avx512::exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), v_dot))));
    
    float p_t;
    _mm_store_ss(&p_t, _mm512_castps512_ps128(v_sig));

    const float threshold = 1.0f - epsilon;
    const float p_clamped = 1.0f - state.p_sum;
    const float exceeds = static_cast<float>(state.p_sum + p_t >= threshold);

    p_t = exceeds * p_clamped + (1.0f - exceeds) * p_t;

    state.p_sum += p_t;
    state.remainder = exceeds * p_t;
    should_halt = (exceeds > 0.5f);
    state.steps += 1;

    return p_t;
}

} // namespace nca::layers
