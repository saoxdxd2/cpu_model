// ============================================================================
// NCA — Spectral Logic Interface Implementation
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
    size_t d_model
) {
    // 1. Transform state and proposal to spectral domain
    alignas(64) float x_spec[2048], y_spec[2048];
    std::copy(state, state + d_model, x_spec);
    std::copy(glr_proposal, glr_proposal + d_model, y_spec);

    nca::spectral::fwht_inplace({x_spec, d_model});
    nca::spectral::fwht_inplace({y_spec, d_model});

    // 2. Perform Kronecker-RLS Update
    rls_state.update(x_spec, y_spec, nca::config::RLS_FORGETTING_FACTOR, 0.01f);

    // 3. Apply updated spectral operator
    alignas(64) float out_spec[2048];
    rls_state.apply(x_spec, out_spec);

    // 4. Inverse transform back to latent space
    nca::spectral::ifwht_inplace({out_spec, d_model});

    // 5. Residual add into state
    for (size_t i = 0; i < d_model; ++i) {
        state[i] += out_spec[i];
    }
}

} // namespace nca::execution
