// ============================================================================
// NCA — Activations Math
// core/activations.cpp
// ============================================================================

#include "core/activations.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/avx512_kernels.hpp"
#include "core/simd/avx2_kernels.hpp"

#include <cmath>
#include <algorithm>

namespace nca::math {

void silu_scalar(float* __restrict data, size_t size) {
    size_t i = 0;
    // 4x Unrolled Scalar SiLU
    for (; i + 3 < size; i += 4) [[likely]] {
        data[i] = data[i] / (1.0f + std::exp(-data[i]));
        data[i + 1] = data[i + 1] / (1.0f + std::exp(-data[i + 1]));
        data[i + 2] = data[i + 2] / (1.0f + std::exp(-data[i + 2]));
        data[i + 3] = data[i + 3] / (1.0f + std::exp(-data[i + 3]));
    }
    for (; i < size; ++i) {
        data[i] = data[i] / (1.0f + std::exp(-data[i]));
    }
}

void silu(std::span<float> data) {
    NCA_DISPATCH_KERNEL(simd::avx512::silu, simd::avx2::silu, silu_scalar, data.data(), data.size());
}

} // namespace nca::math
