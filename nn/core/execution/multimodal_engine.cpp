// ============================================================================
// NCA — Multimodal Engine Implementation (v27.0 - Grounded Breakthrough)
// core/execution/multimodal_engine.cpp
// ============================================================================

#include "core/execution/multimodal_engine.hpp"
#include "config/model_config.hpp"
#include "core/simd/memory.hpp"
#include "core/vision/scanner.hpp"
#include "core/vision/spectral_pruner.hpp"
#include "core/execution/latent_adapter.hpp"
#include "core/layers/glr.hpp"
#include "core/layers/halting.hpp"
#include "core/activations.hpp"
#include "core/spectral/spectral_logic.hpp"
#include "core/spectral/fwht.hpp"
#include <cmath>
#include <algorithm>
#include <span>
#include <random>

namespace nca::execution {

MultimodalEngine::MultimodalEngine(size_t obs_dim, size_t act_dim, nca::config::EngineConfig engine_cfg) 
    : engine_cfg_(engine_cfg), obs_dim_(obs_dim), act_dim_(act_dim), encoder_() {
    const size_t D = nca::config::D_MODEL;
    const size_t n_experts = nca::config::N_MICRO_EXPERTS; 

    // [OPTIMIZATION] Pre-allocate persistent routing buffer
    routing_buffer_.reserve(nca::config::TOP_K_EXPERTS * 2);

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
    std::uniform_real_distribution<float> fog(-1e-6f, 1e-6f);
    for(size_t i=0; i < n_experts; ++i) {
        expert_pool_gate_.emplace_back(D / 32);
        expert_pool_up_.emplace_back(D / 32);
        for(size_t j=0; j<D; ++j) temp[j] = fog(gen); nca::linalg::mx_quantize_w(temp.get(), expert_pool_gate_[i]);
        for(size_t j=0; j<D; ++j) temp[j] = fog(gen); nca::linalg::mx_quantize_w(temp.get(), expert_pool_up_[i]);
    }

    W_halt_ = nca::linalg::MXINT8Tensor(D / 32);
    for(size_t j=0; j<D; ++j) temp[j] = dist(gen); nca::linalg::mx_quantize_w(temp.get(), W_halt_);

    spectral_rls_ = std::make_unique<nca::spectral::KroneckerRLSState>(D);
    state_ = nca::simd::make_aligned_unique<float[]>(D);
    vision_latent_ = nca::simd::make_aligned_unique<float[]>(D);
    h_glr_ = nca::simd::make_aligned_unique<float[]>(D);
    state_momentum_ = nca::simd::make_aligned_unique<float[]>(D);
    prediction_buf_ = nca::simd::make_aligned_unique<float[]>(D);
    
    // [OPTIMIZATION] Pre-allocate batch state buffer
    batch_state_ = nca::simd::make_aligned_unique<float[]>(32 * D); 

    reset_state();
}

void MultimodalEngine::step_batch(const float* text_in, const float* image_in, float* out, size_t batch_size) {
    const size_t D = nca::config::D_MODEL;
    const size_t out_dim = act_dim_ + 1;
    nca::linalg::MXUINT8Tensor x_q(D / 32);

    for (size_t i = 0; i < batch_size; ++i) {
        const float* t_in = text_in ? (text_in + i * D) : nullptr;
        const float* im_in = image_in ? (image_in + i * obs_dim_) : nullptr;
        float* e_out = out + i * out_dim;
        float* env_state = batch_state_.get() + i * D;

        // 1. Perception
        if (im_in && !t_in) {
            encoder_.encode(im_in, env_state, D);
        } else if (t_in) {
            std::memcpy(env_state, t_in, D * sizeof(float));
        }

        // 2. Logic (Single High-Speed Cycle)
        nca::layers::glr_step(env_state, W_glr_alpha_.get(), W_glr_beta_.get(), env_state, D);
        nca::spectral::fwht_inplace({env_state, D});
        
        routing_buffer_.clear();
        router_->route(env_state, routing_buffer_);
        
        nca::linalg::mx_quantize_x(env_state, x_q);
        if (routing_buffer_.size() >= 16) {
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

            for(int j=0; j<16; ++j) {
                float val = (g[j] / (1.0f + std::exp(-g[j]))) * u[j] * 0.1f;
                size_t b_idx = (routing_buffer_[j] * 16) % D;
                for(int k=0; k<16; ++k) env_state[b_idx + k] += val;
            }
        }
        
        nca::spectral::ifwht_no_scale({env_state, D});

        // 3. Actuation
        float sum_sq = 1e-6f;
        for(size_t j=0; j<D; ++j) sum_sq += env_state[j] * env_state[j];
        float rms = std::sqrt(sum_sq / D);
        float scale = 1.0f / (rms + 1e-6f);
        
        size_t write_count = std::min(D, out_dim);
        for(size_t j=0; j<write_count; ++j) e_out[j] = env_state[j] * scale;
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

void MultimodalEngine::reset_state() {
    const size_t D = nca::config::D_MODEL;
    std::memset(state_.get(), 0, D * sizeof(float));
    std::memset(vision_latent_.get(), 0, D * sizeof(float));
    std::memset(h_glr_.get(), 0, D * sizeof(float));
    std::memset(state_momentum_.get(), 0, D * sizeof(float));
    std::memset(prediction_buf_.get(), 0, D * sizeof(float));
    spectral_rls_->reset();
}

void MultimodalEngine::step(const float* text_in, const float* image_in, float* out) {
    const size_t D = nca::config::D_MODEL;
    
    // ── 0. CROSS-MODAL FUSION (ENCODER) ──
    // Simultaneously ingest semantic commands and visual GUI state
    if (image_in) {
        // Encode raw pixels into the vision_latent_ channel
        vision_encoder_.encode_gui(image_in, vision_latent_.get(), D);
        
        // Additive fusion into the primary state wavefront
        for (size_t i = 0; i < D; ++i) state_[i] += vision_latent_[i] * 0.1f;
    }

    if (text_in) {
        // Direct injection of semantic tokens/alphabet primitives
        for (size_t i = 0; i < D; ++i) state_[i] += text_in[i];
    }

    // ── 1. IMPORTANCE CLASSIFICATION ──
    // The engine's "Attention" is now fused across modalities
    ImportanceDecision d = classifier_.classify(state_.get(), state_.get(), prediction_buf_.get(), D);

    // ── 2. STATE INJECTION ──
    if (text_in) {
        if (d.is_fact) {
            // Absolute anchor for facts
            for(size_t i=0; i<D; ++i) state_[i] = text_in[i]; 
        } else {
            // Noise decay
            for(size_t i=0; i<D; ++i) state_[i] *= 0.8f; 
        }
    }

    // [CRITICAL] Grounding Target: Memory should learn the RAW TOKEN for facts.
    alignas(64) float fact_ground[2048];
    if (text_in && d.should_learn) std::copy(text_in, text_in + D, fact_ground);
    else std::copy(state_.get(), state_.get() + D, fact_ground);

    nca::layers::glr_step(h_glr_.get(), W_glr_alpha_.get(), W_glr_beta_.get(), state_.get(), D);
    alignas(64) float glr_snap[2048];
    std::copy(h_glr_.get(), h_glr_.get() + D, glr_snap);

    float accumulated_h = 0.0f;
    nca::layers::HaltingState h_state;
    nca::linalg::MXUINT8Tensor x_q(D / 32);

    // ── 3. TINY RECURSIVE MODELING: H-CYCLES AND L-CYCLES ──
    // H-Cycles (Outer Loop): Answer Revision & Halting evaluation
    // L-Cycles (Inner Loop): Deep latent state refinement (Thought process)
    
    const int L_CYCLES = d.is_fact ? 2 : 1; // Dynamic depth for internal thought
    int h_cycles_executed = 0;

    while (accumulated_h < nca::config::ACT_HALT_THRESHOLD && h_cycles_executed < d.act_cycles) {
        
        bool execute_learning = d.should_learn && (h_cycles_executed == 0);

        // ── L-CYCLES (Inner Thought Loop) ──
        alignas(64) float z_latent[2048];
        std::copy(state_.get(), state_.get() + D, z_latent);

        for (int l = 0; l < L_CYCLES; ++l) {
            if (d.is_fact) {
                // Spectral refinement of thought
                spectral_logic_step(z_latent, fact_ground, *spectral_rls_, D, execute_learning && (l == 0));
            }

            nca::spectral::fwht_inplace({z_latent, D});
            
            std::vector<size_t> active_indices;
            router_->route(z_latent, active_indices);
            nca::linalg::mx_quantize_x(z_latent, x_q);
            float x_norm = nca::linalg::mx_compute_activation_norm(x_q);

            if (active_indices.size() >= 16) {
                const nca::linalg::MXINT8Tensor* gate_ptrs[16];
                const nca::linalg::MXINT8Tensor* up_ptrs[16];
                for(int i=0; i<16; ++i) {
                    size_t id = active_indices[i] % nca::config::N_MICRO_EXPERTS;
                    gate_ptrs[i] = &expert_pool_gate_[id];
                    up_ptrs[i] = &expert_pool_up_[id];
                }

                alignas(64) float g[16], u[16];
                nca::linalg::mx_rank16_dot_ptrs(gate_ptrs, x_q, g);
                nca::linalg::mx_rank16_dot_ptrs(up_ptrs, x_q, u);

                float out_scale = d.is_fact ? 10.0f : 0.01f;
                for(int i=0; i<16; ++i) {
                    float val = (g[i] / (1.0f + std::exp(-g[i]))) * u[i] * out_scale;
                    size_t b_start = (active_indices[i] * 16) % D;
                    for(int j=0; j<16; ++j) {
                        size_t idx = (b_start + j);
                        z_latent[idx] += val; 
                        if (execute_learning && l == 0) {
                            // Ground expert update in the FACT (only once per token)
                            nca::linalg::mx_update_gaussian_moment(const_cast<nca::linalg::MXINT8Tensor&>(*gate_ptrs[i]), x_q, fact_ground[idx] - z_latent[idx], 10.0f, x_norm);
                            nca::linalg::mx_update_gaussian_moment(const_cast<nca::linalg::MXINT8Tensor&>(*up_ptrs[i]), x_q, fact_ground[idx] - z_latent[idx], 10.0f, x_norm);
                        }
                    }
                }
            }
            nca::spectral::ifwht_no_scale({z_latent, D});
            for(size_t i=0; i<D; ++i) {
                if(!std::isfinite(z_latent[i])) z_latent[i] = 0.0f;
                z_latent[i] = std::clamp(z_latent[i], -10.0f, 10.0f);
            }
        } // End L-Cycle

        // ── H-CYCLE (Outer Answer Revision) ──
        // Update the main state with the refined thought
        std::copy(z_latent, z_latent + D, state_.get());

        nca::linalg::mx_quantize_x(state_.get(), x_q);
        bool halt = false;
        accumulated_h += nca::layers::halting_step(x_q, W_halt_, 0.0f, h_state, halt);
        if(halt && !d.is_fact) break; 
        
        h_cycles_executed++;
    }

    
    std::copy(state_.get(), state_.get() + D, prediction_buf_.get());

    if (out) {
        float sum_sq = 0.0f;
        for(size_t i=0; i<D; ++i) sum_sq += state_[i] * state_[i];
        float rms = std::sqrt(sum_sq / D);
        float scale = (rms > 0.5f) ? (1.0f / (rms + 1e-6f)) : 2.0f;
        
        // Only write up to act_dim + 1 (Value)
        size_t write_count = std::min(D, act_dim_ + 1);
        for(size_t i=0; i<write_count; ++i) out[i] = state_[i] * scale;
    }
}

void MultimodalEngine::update_from_trajectory(size_t count, size_t obs_dim, size_t act_dim, 
                                              const float* states, const float* actions, const float* advantages) {
    const size_t D = nca::config::D_MODEL;
    alignas(64) float padded_state[2048];
    
    for (size_t t = 0; t < count; ++t) {
        const float* state = states + t * obs_dim;
        float adv = std::clamp(advantages[t], -10.0f, 10.0f); // [STABILITY] Advantage Clipping
        std::memset(padded_state, 0, D * sizeof(float));
        std::memcpy(padded_state, state, std::min(obs_dim, D) * sizeof(float));

        // 1. Evaluate novelty using the engine's classifier
        ImportanceDecision d = classifier_.classify(padded_state, state_.get(), prediction_buf_.get(), D);
        
        // 2. If it's a novel fact or requires learning, connect RL Advantage to NN updates
        if (d.should_learn) {
            nca::linalg::MXUINT8Tensor x_q(D / 32);
            nca::linalg::mx_quantize_x(padded_state, x_q);
            float x_norm = nca::linalg::mx_compute_activation_norm(x_q);

            std::vector<size_t> active_indices;
            router_->route(padded_state, active_indices);

            // Backpropagate the advantage directly into the active micro-experts
            if (active_indices.size() >= 16) {
                for(int i=0; i<16; ++i) {
                    size_t id = active_indices[i] % nca::config::N_MICRO_EXPERTS;
                    
                    // Gaussian Moment Update driven by GAE!
                    nca::linalg::mx_update_gaussian_moment(expert_pool_gate_[id], x_q, adv, 1.0f, x_norm);
                    nca::linalg::mx_update_gaussian_moment(expert_pool_up_[id], x_q, adv, 1.0f, x_norm);
                }
            }
        }
    }
}

} // namespace nca::execution
