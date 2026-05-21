#pragma once
// ============================================================================
// NCA — E-AdaPrune Spectral Pruner
// core/vision/spectral_pruner.hpp
// ============================================================================

#include <cstddef>
#include <vector>
#include <span>

namespace nca::vision {

// ── SPECTRAL PRUNER (Phase 11) ──────────────────────────────────────────────
// Uses a SIMD Power Iteration to find high-variance patches.
// Destroys complexity by replacing O(N^3) SVD with O(N) estimation.
class SpectralPruner {
public:
    struct Config {
        size_t n_tokens;
        size_t d_model;
        float keep_ratio; // Percentage of tokens to keep
    };

    explicit SpectralPruner(Config cfg);

    // Determines which tokens to keep based on spectral variance.
    // Returns the number of kept tokens.
    size_t prune(
        std::span<const float> x_patches,
        std::span<size_t> out_active_indices
    ) const;

private:
    Config cfg_;
};

} // namespace nca::vision
