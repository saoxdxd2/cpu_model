// ============================================================================
// NCA -- Halting Gate (Phase 7)
// core/layers/halting.cpp
// ============================================================================

#include "core/layers/halting.hpp"
#include "core/simd/avx512_math.hpp"
#include <cmath>
#include <algorithm>

namespace nca::layers {

float halting_step(const nca::linalg::MXUINT8Tensor& x_q, const nca::linalg::MXINT8Tensor& w_halt, float b_halt, HaltingState& state, bool& should_halt) {
    float dot = nca::linalg::mx_dot(w_halt, x_q) + b_halt;

    // ── ALGEBRAIC PROXY (Target 3) ──────────────────────────────────────────
    // Instead of Shannon Entropy (-p log p), we use the geometric proxy 
    // Gini Impurity or a fast Sigmoid-based decision.
    // Here, we use the branchless minimax sigmoid.
    alignas(64) float buf[16] = {dot};
    auto v_sig = _mm512_rcp14_ps(_mm512_add_ps(_mm512_set1_ps(1.f), nca::simd::avx512::exp_ps(_mm512_sub_ps(_mm512_setzero_ps(), _mm512_load_ps(buf)))));
    float pt; _mm_store_ss(&pt, _mm512_castps512_ps128(v_sig));

    const float thresh = 0.99f, pc = 1.f - state.p_sum;
    const float ex = (float)(state.p_sum + pt >= thresh);
    pt = ex * pc + (1.f - ex) * pt;
    state.p_sum += pt; state.remainder = ex * pt; should_halt = (ex > .5f); state.steps++;
    return pt;
}

} // namespace nca::layers
