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

    expert_pool_gate.reserve(n_experts);
    expert_pool_up.reserve(n_experts);
    auto temp = nca::simd::make_aligned_unique<float[]>(d_model);
    for(size_t i=0; i < n_experts; ++i) {
        expert_pool_gate.emplace_back(d_model / 32);
        expert_pool_up.emplace_back(d_model / 32);
        for(size_t j=0; j<d_model; ++j) temp[j] = dist(gen); nca::linalg::mx_quantize_w(temp.get(), expert_pool_gate[i]);
        for(size_t j=0; j<d_model; ++j) temp[j] = dist(gen); nca::linalg::mx_quantize_w(temp.get(), expert_pool_up[i]);
    }

    halting_gate = nca::linalg::MXINT8Tensor(d_model / 32);
    for(size_t j=0; j<d_model; ++j) temp[j] = dist(gen); nca::linalg::mx_quantize_w(temp.get(), halting_gate);
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
