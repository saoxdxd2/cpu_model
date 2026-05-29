#pragma once
#include "core/simd/memory.hpp"
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

    // Legacy MoE / MX dependencies have been removed per Geometric Schema Migration
    
    // Binary Curve Tree (Transistor-Level Weights)
    size_t bct_m = 0;
    size_t bct_n = 0;
    ::nca::simd::aligned_unique_ptr<uint16_t[]> binary_curve_masks;
    float bct_scale = 1.0f;
    float bct_offset = 0.0f;

    void initialize_unit_noise(size_t d_model, size_t n_experts);
    void initialize_binary_curve(size_t m, size_t n);
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
