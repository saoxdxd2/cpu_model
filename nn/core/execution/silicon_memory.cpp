#include "nn/core/execution/silicon_memory.hpp"
#include <random>
#include <cstring>

namespace nca::execution {

void SiliconWeights::initialize_unit_noise(size_t d_model, size_t n_experts) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

    vision_A = nca::simd::make_aligned_unique<float[]>(16 * 16 * 128 * 16);
    vision_B = nca::simd::make_aligned_unique<float[]>(16);
    vision_C = nca::simd::make_aligned_unique<float[]>(16);
    for(size_t i=0; i<16*16*128*16; ++i) vision_A[i] = dist(gen);
    for(size_t i=0; i<16; ++i) { vision_B[i] = dist(gen); vision_C[i] = dist(gen); }

    glr_alpha = nca::simd::make_aligned_unique<float[]>(d_model);
    glr_beta = nca::simd::make_aligned_unique<float[]>(d_model);
    for(size_t i=0; i<d_model; ++i) { 
        glr_alpha[i] = 0.999f + dist(gen) * 0.0001f; 
        glr_beta[i] = 0.1f + dist(gen) * 0.01f;
    }

    // Legacy MoE / MX dependencies have been removed per Geometric Schema Migration
}

void SiliconWeights::initialize_binary_curve(size_t m, size_t n) {
    bct_m = m;
    bct_n = n;
    size_t num_masks = m * (n / 16);
    binary_curve_masks = nca::simd::make_aligned_unique<uint16_t[]>(num_masks);
    
    std::mt19937 gen(1337);
    std::uniform_int_distribution<uint16_t> dist(0, 65535);
    for (size_t i = 0; i < num_masks; ++i) {
        binary_curve_masks[i] = dist(gen);
    }
    bct_scale = 1.0f;
    bct_offset = 0.0f;
}

SiliconWavefront::SiliconWavefront(size_t d_model) {
    state = nca::simd::make_aligned_unique<float[]>(d_model);
    momentum = nca::simd::make_aligned_unique<float[]>(d_model);
    h_glr = nca::simd::make_aligned_unique<float[]>(d_model);
    prediction_buf = nca::simd::make_aligned_unique<float[]>(d_model);
    reset(d_model);
}

void SiliconWavefront::reset(size_t d_model) {
    std::memset(state.get(), 0, d_model * sizeof(float));
    std::memset(momentum.get(), 0, d_model * sizeof(float));
    std::memset(h_glr.get(), 0, d_model * sizeof(float));
    std::memset(prediction_buf.get(), 0, d_model * sizeof(float));
}

} // namespace nca::execution
