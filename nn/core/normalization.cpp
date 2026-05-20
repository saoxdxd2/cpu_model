// ============================================================================
// NCA — High-Precision Normalization Math
// core/normalization.cpp
//
// Runtime dispatcher for RMSNorm.
// ============================================================================

#include "core/normalization.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/avx2_kernels.hpp"
#include "core/simd/avx512_kernels.hpp"

#include <cmath>

namespace nca::math {

// Fallback scalar implementation
void rmsnorm_scalar(float* out, const float* in, const float* weight, int size, float eps) {
    float sq_sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sq_sum += in[i] * in[i];
    }
    float inv_rms = 1.0f / std::sqrt(sq_sum / size + eps);
    for (int i = 0; i < size; ++i) {
        out[i] = in[i] * inv_rms * weight[i];
    }
}

void rmsnorm(float* out, const float* in, const float* weight, int size, float eps) {
    using simd::Backend;
    
    // Note: To avoid branching on every single layer call, we could cache the 
    // backend locally, or the engine could orchestrate the dispatch directly.
    // For now, best_backend() reads a cached static variable, so it's ~1ns.
    Backend b = simd::best_backend();
    
    if (b == Backend::AVX512) {
        simd::avx512::rmsnorm(out, in, weight, size, eps);
    } else if (b == Backend::AVX2) {
        simd::avx2::rmsnorm(out, in, weight, size, eps);
    } else {
        rmsnorm_scalar(out, in, weight, size, eps);
    }
}

} // namespace nca::math
