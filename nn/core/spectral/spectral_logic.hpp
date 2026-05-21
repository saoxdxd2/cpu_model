#pragma once
// ============================================================================
// NCA — Spectral Logic Interface
// core/spectral/spectral_logic.hpp
// ============================================================================

#include "core/spectral/kronecker_rls.hpp"
#include <cstddef>

namespace nca::execution {

// ── SPECTRAL LOGIC STEP (Target 2) ──────────────────────────────────────────
// Plugs into the ACT loop. Transforms to frequency domain, updates RLS state,
// and applies spectral correction.
void spectral_logic_step(
    float* state, 
    const float* glr_proposal,
    nca::spectral::KroneckerRLSState& rls_state,
    size_t d_model
);

} // namespace nca::execution
