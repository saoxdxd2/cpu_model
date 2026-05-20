#pragma once
// ============================================================================
// NCA — AVX-512 VNNI Kernels
// core/simd/avx512_vnni.hpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"

namespace nca::simd::avx512 {

// VNNI _mm512_dpbusd_epi32 exact dot product kernel
float vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w, const nca::linalg::MXUINT8Tensor* __restrict x);

} // namespace nca::simd::avx512
