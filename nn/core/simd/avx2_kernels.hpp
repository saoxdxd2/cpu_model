#pragma once
#include <span>

namespace nca::simd::avx2 {

void rmsnorm(float* __restrict out, const float* __restrict in, const float* __restrict weight, size_t size, float eps);

} // namespace nca::simd::avx2
