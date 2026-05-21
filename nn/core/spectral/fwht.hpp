#pragma once
// ============================================================================
// NCA — Fast Walsh-Hadamard Transform (FWHT)
// core/spectral/fwht.hpp
// ============================================================================

#include <span>
#include <cstddef>

namespace nca::spectral {

// Performs an in-place Fast Walsh-Hadamard Transform.
// Complexity: O(N log N).
// input: Span of floats, size must be power of 2.
void fwht_inplace(std::span<float> data);

// Performs an in-place Inverse Fast Walsh-Hadamard Transform.
// Note: FWHT is its own inverse up to a scaling factor 1/N.
void ifwht_inplace(std::span<float> data);

// Inverse transform without the 1/N scaling factor.
void ifwht_no_scale(std::span<float> data);

} // namespace nca::spectral
