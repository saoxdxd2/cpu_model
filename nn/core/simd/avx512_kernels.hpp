#pragma once
#include <span>

namespace nca::simd::avx512 {

// Note the use of __restrict to guarantee to the compiler that memory never overlaps.
// This enables aggressive instruction reordering and caching.
void rmsnorm(float* __restrict out, const float* __restrict in, const float* __restrict weight, size_t size, float eps);
void silu(float* __restrict data, size_t size);

} // namespace nca::simd::avx512
