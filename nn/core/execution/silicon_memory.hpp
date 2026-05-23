#pragma once
#include "core/simd/memory.hpp"
#include "core/linalg/mx_linear.hpp"
#include <vector>

namespace nca::execution {

/**
 * SiliconWeights
 * Centralized registry for all hardware-saturated weights.
 */
struct SiliconWeights {
    // Vision Primitives
    ::nca::simd::aligned_unique_ptr<float[]> vision_A;
    ::nca::simd::aligned_unique_ptr<float[]> vision_B;
    ::nca::simd::aligned_unique_ptr<float[]> vision_C;

    // Recurrence Factors
    ::nca::simd::aligned_unique_ptr<float[]> glr_alpha;
    ::nca::simd::aligned_unique_ptr<float[]> glr_beta;

    // SDMS Expert Pool
    std::vector<::nca::linalg::MXINT8Tensor> expert_pool_gate;
    std::vector<::nca::linalg::MXINT8Tensor> expert_pool_up;

    // Control Gates
    ::nca::linalg::MXINT8Tensor halting_gate;

    void initialize_unit_noise(size_t d_model, size_t n_experts);
};

/**
 * SiliconWavefront
 * Isolated mental state for a single agent session.
 */
struct SiliconWavefront {
    ::nca::simd::aligned_unique_ptr<float[]> state;
    ::nca::simd::aligned_unique_ptr<float[]> momentum;
    ::nca::simd::aligned_unique_ptr<float[]> h_glr;
    ::nca::simd::aligned_unique_ptr<float[]> prediction_buf;

    SiliconWavefront(size_t d_model);
    void reset(size_t d_model);
};

} // namespace nca::execution
