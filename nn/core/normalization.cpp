// ============================================================================
// NCA — High-Precision Normalization Math
// core/normalization.cpp
// ============================================================================

#include "core/normalization.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/avx2_kernels.hpp"
#include "core/simd/avx512_kernels.hpp"

#include <cmath>
#include <stdexcept>

namespace nca::math {

void rmsnorm_scalar(float* __restrict out, const float* __restrict in, const float* __restrict weight, size_t size, float eps) {
    float sq_sum = 0.0f;
    size_t i = 0;
    // 8x Unrolled Scalar Square Sum
    for (; i + 7 < size; i += 8) [[likely]] {
        sq_sum += in[i] * in[i];
        sq_sum += in[i + 1] * in[i + 1];
        sq_sum += in[i + 2] * in[i + 2];
        sq_sum += in[i + 3] * in[i + 3];
        sq_sum += in[i + 4] * in[i + 4];
        sq_sum += in[i + 5] * in[i + 5];
        sq_sum += in[i + 6] * in[i + 6];
        sq_sum += in[i + 7] * in[i + 7];
    }
    for (; i < size; ++i) sq_sum += in[i] * in[i];
    
    auto inv_rms = 1.0f / std::sqrt(sq_sum / size + eps);
    
    i = 0;
    // 8x Unrolled Scalar Normalization
    for (; i + 7 < size; i += 8) [[likely]] {
        out[i] = in[i] * inv_rms * weight[i];
        out[i + 1] = in[i + 1] * inv_rms * weight[i + 1];
        out[i + 2] = in[i + 2] * inv_rms * weight[i + 2];
        out[i + 3] = in[i + 3] * inv_rms * weight[i + 3];
        out[i + 4] = in[i + 4] * inv_rms * weight[i + 4];
        out[i + 5] = in[i + 5] * inv_rms * weight[i + 5];
        out[i + 6] = in[i + 6] * inv_rms * weight[i + 6];
        out[i + 7] = in[i + 7] * inv_rms * weight[i + 7];
    }
    for (; i < size; ++i) out[i] = in[i] * inv_rms * weight[i];
}

void rmsnorm(std::span<float> out, std::span<const float> in, std::span<const float> weight, float eps) {
    if (out.size() != in.size() || in.size() != weight.size()) [[unlikely]] {
        throw std::invalid_argument("rmsnorm: Tensor spans must have equal size");
    }

    NCA_DISPATCH_KERNEL(simd::avx512::rmsnorm, simd::avx2::rmsnorm, rmsnorm_scalar, out.data(), in.data(), weight.data(), in.size(), eps);
}

} // namespace nca::math
