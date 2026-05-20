#pragma once
// ============================================================================
// NCA — Activations Math
// core/activations.hpp
// ============================================================================

#include <span>

namespace nca::math {

/// Applies SiLU (Swish) Activation function in-place.
/// Automatically dispatches to the most optimal SIMD backend.
/// @param data Tensor buffer to activate
void silu(std::span<float> data);

} // namespace nca::math
