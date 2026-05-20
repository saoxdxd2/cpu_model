#pragma once
// ============================================================================
// NCA — AVX-512 VNNI Kernels
// core/simd/avx512_vnni.hpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"

namespace nca::simd::avx512 {

// Computes dot product between E8M0 int8 weights and uint8 activations using VNNI instructions
float vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w, const nca::linalg::MXUINT8Tensor* __restrict x);

void mx_quantize_x(const float* __restrict in, nca::linalg::MXUINT8Tensor* __restrict out);

} // namespace nca::simd::avx512
