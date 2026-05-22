// ============================================================================
// NCA — Multimodal Engine Implementation (v30.0 - Grand Silicon Hardened)
// core/execution/multimodal_engine.cpp
// ============================================================================

#include "core/execution/multimodal_engine.hpp"
#include "core/simd/avx512_math.hpp"
#include "core/spectral/fwht.hpp"
#include "core/layers/glr.hpp"
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>

namespace nca::execution {

MultimodalEngine::MultimodalEngine(size_t obs_dim, size_t act_dim, nca::config::EngineConfig engine_cfg) 
    : engine_cfg_(engine_cfg), obs_dim_(obs_dim), act_dim_(act_dim) {
    
    const size_t D = nca::config::D_MODEL;
    const size_t n_experts = nca::config::N_MICRO_EXPERTS; 

    // [OPTIMIZATION] Pre-allocate all buffers (128 Batch Limit for Zero-Allocation)
    batch_state_ = nca::simd::make_aligned_unique<float[]>(128 * D); 
    state_ = nca::simd::make_aligned_unique<float[]>(D);
    vision_latent_ = nca::simd::make_aligned_unique<float[]>(D);
    h_glr_ = nca::simd::make_aligned_unique<float[]>(D);
    state_momentum_ = nca::simd::make_aligned_unique<float[]>(D);
    prediction_buf_ = nca::simd::make_aligned_unique<float[]>(D);

    // Initialize Weights
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

    W_vision_A_ = nca::simd::make_aligned_unique<float[]>(16 * 16 * 128 * 16);
    W_vision_B_ = nca::simd::make_aligned_unique<float[]>(16);
    W_vision_C_ = nca::simd::make_aligned_unique<float[]>(16);
    for(size_t i=0; i<16*16*128*16; ++i) W_vision_A_[i] = dist(gen);
    for(size_t i=0; i<16; ++i) { W_vision_B_[i] = dist(gen); W_vision_C_[i] = dist(gen); }

    W_glr_alpha_ = nca::simd::make_aligned_unique<float[]>(D);
    W_glr_beta_ = nca::simd::make_aligned_unique<float[]>(D);
    for(size_t i=0; i<D; ++i) { 
        W_glr_alpha_[i] = 0.999f + dist(gen) * 0.0001f; 
        W_glr_beta_[i] = 0.1f + dist(gen) * 0.01f;
    }

    router_ = std::make_unique<nca::linalg::HashedRouter>(nca::linalg::HashedRouter::Config{D, n_experts, nca::config::TOP_K_EXPERTS, 32});

    expert_pool_gate_.reserve(n_experts);
    expert_pool_up_.reserve(n_experts);
    auto temp = nca::simd::make_aligned_unique<float[]>(D);
    for(size_t i=0; i < n_experts; ++i) {
        expert_pool_gate_.emplace_back(D / 32);
        expert_pool_up_.emplace_back(D / 32);
        for(size_t j=0; j<D; ++j) temp[j] = dist(gen); nca::linalg::mx_quantize_w(temp.get(), expert_pool_gate_[i]);
        for(size_t j=0; j<D; ++j) temp[j] = dist(gen); nca::linalg::mx_quantize_w(temp.get(), expert_pool_up_[i]);
    }

    W_halt_ = nca::linalg::MXINT8Tensor(D / 32);
    for(size_t j=0; j<D; ++j) temp[j] = dist(gen); nca::linalg::mx_quantize_w(temp.get(), W_halt_);

    spectral_rls_ = std::make_unique<nca::spectral::KroneckerRLSState>(D);
    reset_state();
}

void MultimodalEngine::reset_state() {
    const size_t D = nca::config::D_MODEL;
    std::memset(state_.get(), 0, D * sizeof(float));
    std::memset(vision_latent_.get(), 0, D * sizeof(float));
    std::memset(h_glr_.get(), 0, D * sizeof(float));
    std::memset(state_momentum_.get(), 0, D * sizeof(float));
    std::memset(prediction_buf_.get(), 0, D * sizeof(float));
    spectral_rls_->reset();
}

void MultimodalEngine::step_batch(const float* text_in, const float* image_in, float* out, size_t batch_size) {
    const size_t D = nca::config::D_MODEL;
    const size_t out_dim = act_dim_ + 1;
    nca::linalg::MXUINT8Tensor x_q(D / 32);
    
    // Safety check for pre-allocated buffer
    if (batch_size > 128) batch_size = 128;

    for (size_t b = 0; b < batch_size; ++b) {
        float* env_state = (batch_size == 1) ? state_.get() : (batch_state_.get() + b * D);
        const float* t_in = text_in ? (text_in + b * D) : nullptr;
        const float* im_in = image_in ? (image_in + b * obs_dim_) : nullptr;
        float* e_out = out + b * out_dim;

        // ── 1. CROSS-MODAL FUSION (Silicon-Level Gating) ──
        if (im_in) {
            vision_encoder_.encode_gui(im_in, vision_latent_.get(), D);

            // [HARDENED] Use Text as a Saliency Gate for Vision
            if (t_in) {
                for (size_t i = 0; i < D; ++i) {
                    // Alpha-Vision Cross-Gating
                    float gate = 1.0f + std::tanh(t_in[i]);
                    env_state[i] += vision_latent_[i] * gate * 0.1f;
                    env_state[i] = std::clamp(env_state[i], -10.0f, 10.0f);
                }
            } else {
                for (size_t i = 0; i < D; ++i) {
                    env_state[i] += vision_latent_[i] * 0.1f;
                    env_state[i] = std::clamp(env_state[i], -10.0f, 10.0f);
                }
            }
        }
        if (t_in) {
            // Base alphabet injection
            for (size_t i = 0; i < D; ++i) {
                env_state[i] += t_in[i];
                env_state[i] = std::clamp(env_state[i], -10.0f, 10.0f);
            }
        }

        // ── 2. SILICON REASONING (Hardened) ──
        nca::layers::glr_step(env_state, W_glr_alpha_.get(), W_glr_beta_.get(), env_state, D);
        nca::spectral::fwht_inplace({env_state, D});
        
        size_t r_count = 0;
        router_->route_to_buffer(env_state, routing_buffer_, &r_count);
        
        nca::linalg::mx_quantize_x(env_state, x_q);
        if (r_count >= 16) {
            const nca::linalg::MXINT8Tensor* gate_ptrs[16];
            const nca::linalg::MXINT8Tensor* up_ptrs[16];
            for(int j=0; j<16; ++j) {
                size_t id = routing_buffer_[j] % nca::config::N_MICRO_EXPERTS;
                gate_ptrs[j] = &expert_pool_gate_[id];
                up_ptrs[j] = &expert_pool_up_[id];
            }
            alignas(64) float g[16], u[16];
            nca::linalg::mx_rank16_dot_ptrs(gate_ptrs, x_q, g);
            nca::linalg::mx_rank16_dot_ptrs(up_ptrs, x_q, u);

            __m512 vG = _mm512_load_ps(g);
            __m512 vU = _mm512_load_ps(u);
            __m512 vRes = _mm512_mul_ps(nca::simd::avx512::silu_ps(vG), vU);
            _mm512_store_ps(g, vRes);

            for(int j=0; j<16; ++j) {
                size_t b_idx = (routing_buffer_[j] * 16) % D;
                for(int k=0; k<16; ++k) env_state[b_idx + k] += g[j] * 0.1f;
            }
        }
        
        nca::spectral::ifwht_no_scale({env_state, D});

        // ── 3. ACTUATION ──
        float sum_sq = 1e-8f; 
        for(size_t j=0; j<D; ++j) sum_sq += env_state[j] * env_state[j];
        float rms = std::sqrt(sum_sq / D);
        float scale = 1.0f / (rms + 1e-6f);
        scale = std::clamp(scale, 0.001f, 100.0f);
        
        size_t write_count = std::min(D, out_dim);
        for(size_t j=0; j<write_count; ++j) e_out[j] = env_state[j] * scale;
    }
}

void MultimodalEngine::update_from_trajectory(size_t count, size_t obs_dim, size_t act_dim, 
                                              const float* states, const float* actions, const float* advantages) {
    const size_t D = nca::config::D_MODEL;
    nca::linalg::MXUINT8Tensor x_q(D / 32);
    
    for (size_t t = 0; t < count; ++t) {
        const float* state = states + t * obs_dim;
        float adv = std::clamp(advantages[t], -5.0f, 5.0f); 

        nca::linalg::mx_quantize_x(state, x_q);
        float x_norm = nca::linalg::mx_compute_activation_norm(x_q);

        size_t r_count = 0;
        router_->route_to_buffer(state, routing_buffer_, &r_count);

        if (r_count >= 16) {
            for(int i=0; i<16; ++i) {
                size_t id = routing_buffer_[i] % nca::config::N_MICRO_EXPERTS;
                nca::linalg::mx_update_gaussian_moment(expert_pool_gate_[id], x_q, adv, 1e-5f, x_norm);
                nca::linalg::mx_update_gaussian_moment(expert_pool_up_[id], x_q, adv, 1e-5f, x_norm);
            }
        }
    }
}

WeightRegistry MultimodalEngine::get_weight_registry() {
    WeightRegistry reg;
    const size_t n_experts = nca::config::N_MICRO_EXPERTS;
    reg.expert_pool_gate.reserve(n_experts);
    reg.expert_pool_up.reserve(n_experts);
    for(size_t i=0; i<n_experts; ++i) {
        reg.expert_pool_gate.push_back(&expert_pool_gate_[i]);
        reg.expert_pool_up.push_back(&expert_pool_up_[i]);
    }
    reg.halting_gate = &W_halt_;
    reg.vision_A = W_vision_A_.get();
    reg.vision_B = W_vision_B_.get();
    reg.vision_C = W_vision_C_.get();
    reg.glr_alpha = W_glr_alpha_.get();
    reg.glr_beta = W_glr_beta_.get();
    reg.d_model = nca::config::D_MODEL;
    reg.n_experts = nca::config::N_MICRO_EXPERTS;
    return reg;
}

} // namespace nca::execution
