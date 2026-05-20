#pragma once
// ============================================================================
// NCA — GLR Backbone Layer
// core/layers/glr.hpp
// ============================================================================

#include <cstddef>

namespace nca::layers {

// Performs a single temporal step of the Gated Linear RNN.
// Equation: h_t = alpha * h_{t-1} + beta * x_t
void glr_step(float* h, const float* alpha, const float* beta, const float* x, size_t d_size);

} // namespace nca::layers
