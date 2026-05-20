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
    for (size_t i = 0; i < size; ++i) [[likely]] {
        sq_sum += in[i] * in[i];
    }
    
    // For scalar we still use standard sqrt as rsqrt instruction isn't exposed easily without intrinsics
    float inv_rms = 1.0f / std::sqrt(sq_sum / size + eps);
    for (size_t i = 0; i < size; ++i) [[likely]] {
        out[i] = in[i] * inv_rms * weight[i];
    }
}

void rmsnorm(std::span<float> out, std::span<const float> in, std::span<const float> weight, float eps) {
    if (out.size() != in.size() || in.size() != weight.size()) [[unlikely]] {
        throw std::invalid_argument("rmsnorm: Tensor spans must have equal size");
    }

    NCA_DISPATCH_KERNEL(simd::avx512::rmsnorm, simd::avx2::rmsnorm, rmsnorm_scalar, out.data(), in.data(), weight.data(), in.size(), eps);
}

} // namespace nca::math
