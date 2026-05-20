#pragma once
// ============================================================================
// NCA — High-Precision Normalization Math
// core/normalization.hpp
//
// Utilizing C++20 concepts and std::span for robust, bound-safe passing.
// ============================================================================

#include <span>
#include <concepts>

namespace nca::math {

/// Applies Root Mean Square Normalization (RMSNorm) over the last dimension.
/// Automatically dispatches to AVX-512, AVX2, or Scalar backend at runtime.
/// @param out    Output buffer
/// @param in     Input buffer 
/// @param weight Learnable weight vector
/// @param eps    Epsilon for numerical stability
void rmsnorm(std::span<float> out, std::span<const float> in, std::span<const float> weight, float eps = 1e-5f);

} // namespace nca::math
