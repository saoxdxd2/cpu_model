#pragma once
// ============================================================================
// NCA — AVX2 Math Kernels
// core/simd/avx2_kernels.hpp
// ============================================================================

namespace nca::simd::avx2 {

void rmsnorm(float* out, const float* in, const float* weight, int size, float eps);

} // namespace nca::simd::avx2
