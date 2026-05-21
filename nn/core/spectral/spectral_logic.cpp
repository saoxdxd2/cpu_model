// ============================================================================
// NCA — Spectral Logic Interface Implementation (v3.0 - Evidence Conserving)
// core/spectral/spectral_logic.cpp
// ============================================================================

#include "core/spectral/spectral_logic.hpp"
#include "core/spectral/fwht.hpp"
#include "config/model_config.hpp"
#include <vector>
#include <algorithm>

namespace nca::execution {

void spectral_logic_step(
    float* state, 
    const float* glr_proposal,
    nca::spectral::KroneckerRLSState& rls_state,
    size_t d_model,
    bool should_learn
) {
    // 1. Transform state to spectral domain
    alignas(64) float x_spec[2048];
    std::copy(state, state + d_model, x_spec);
    nca::spectral::fwht_inplace({x_spec, d_model});

    if (should_learn) {
        alignas(64) float y_spec[2048];
        std::copy(glr_proposal, glr_proposal + d_model, y_spec);
        nca::spectral::fwht_inplace({y_spec, d_model});
        rls_state.update(x_spec, y_spec, nca::config::RLS_FORGETTING_FACTOR, 0.01f);
    }

    // 2. Apply spectral operator (Recall or Correct)
    alignas(64) float out_spec[2048];
    rls_state.apply(x_spec, out_spec);

    // 3. Inverse transform back to latent space
    nca::spectral::ifwht_inplace({out_spec, d_model});

    // 4. [FIX] Additive Residual: Preserve the input signal anchor
    for (size_t i = 0; i < d_model; ++i) {
        state[i] += out_spec[i] * 0.1f;
    }
}

} // namespace nca::execution
