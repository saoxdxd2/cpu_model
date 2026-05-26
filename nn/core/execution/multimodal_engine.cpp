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

    // 1. Initialize Silicon Registry
    weights_.initialize_unit_noise(D, n_experts);
    primary_wavefront_ = std::make_unique<SiliconWavefront>(D);
    
    // 2. Pre-allocate Batch Pool
    batch_state_pool_ = nca::simd::make_aligned_unique<float[]>(128 * D); 

    router_ = std::make_unique<nca::linalg::HashedRouter>(nca::linalg::HashedRouter::Config{D, n_experts, nca::config::TOP_K_EXPERTS, 32});
    spectral_rls_ = std::make_unique<nca::spectral::KroneckerRLSState>(D);

    // [GEOMETRIC] Initialize the Wavefront Router (dormant until a .geo graph is loaded)
    geometric_router_ = std::make_unique<WavefrontRouter>(D, 16);
}

void MultimodalEngine::reset_state() {
    primary_wavefront_->reset(nca::config::D_MODEL);
    spectral_rls_->reset();
}

void MultimodalEngine::step(const float* text_in, const float* image_in, float* out) {
    step_batch(text_in, image_in, out, 1);
}

void MultimodalEngine::step_swarm(const float* initial_input, float* swarm_out, size_t max_agents) {
    const size_t D = nca::config::D_MODEL;
    const size_t out_dim = act_dim_ + 1;
    
    if (max_agents > 128) max_agents = 128;

    float* current_context = const_cast<float*>(initial_input);
    float accumulated_h = 0.0f;
    size_t active_swarm_size = 0;

    for (size_t n = 0; n < max_agents; ++n) {
        float* agent_state = batch_state_pool_.get() + n * D;
        float* agent_out = swarm_out + n * out_dim;
        
        // ── 1. CHAINED INGESTION ──
        std::memset(agent_state, 0, D * sizeof(float));
        for (size_t i = 0; i < D; ++i) agent_state[i] += current_context[i];

        // ── 2. RECURSIVE THOUGHT CYCLE (Isolated via Pool) ──
        // We pass agent_state as the primary state for this step
        step_batch(agent_state, nullptr, agent_out, 1);

        // ── 3. ADAPTIVE HALTING (Silicon-ACT) ──
        float h = 1.0f / (1.0f + std::exp(-agent_out[0]));
        accumulated_h += h;
        active_swarm_size++;

        if (accumulated_h > nca::config::ACT_HALT_THRESHOLD) break;
        current_context = agent_state; 
    }

    std::cout << "  [SWARM] Wavefront reached consensus with " << active_swarm_size << " agents.\n";
}

void MultimodalEngine::step_batch(const float* text_in, const float* image_in, float* out, size_t batch_size) {
    const size_t D = nca::config::D_MODEL;
    const size_t out_dim = act_dim_ + 1;
    nca::linalg::MXUINT8Tensor x_q(D / 32);
    
    if (batch_size > 128) batch_size = 128;

    for (size_t b = 0; b < batch_size; ++b) {
        // [BUGFIX] Correct wavefront isolation logic
        // If text_in is null, we assume we're in standalone mode using primary wavefront.
        // If text_in is provided, we use the batch pool or the provided pointer.
        float* env_state = (batch_size == 1 && text_in == nullptr) ? primary_wavefront_->state.get() : 
                          (text_in ? (float*)text_in + b * D : (batch_state_pool_.get() + b * D));
        
        const float* im_in = image_in ? (image_in + b * obs_dim_) : nullptr;
        const float* t_in = text_in ? (text_in + b * D) : nullptr;
        float* e_out = out + b * out_dim;

        // ── 1. CROSS-MODAL FUSION (Silicon-Level Gating) ──
        if (im_in) {
            vision_encoder_.encode_gui(im_in, primary_wavefront_->prediction_buf.get(), D);
            float gate = t_in ? (1.0f + std::tanh(t_in[0])) : 0.1f;
            // [ILP] Branchless AVX-512 fused FMA + clamp
            __m512 v_gate = _mm512_set1_ps(gate);
            __m512 v_hi = _mm512_set1_ps(10.0f), v_lo = _mm512_set1_ps(-10.0f);
            for (size_t i = 0; i < D; i += 16) {
                __m512 v_s = _mm512_loadu_ps(env_state + i);
                __m512 v_p = _mm512_loadu_ps(primary_wavefront_->prediction_buf.get() + i);
                v_s = _mm512_fmadd_ps(v_p, v_gate, v_s);
                _mm512_storeu_ps(env_state + i, _mm512_min_ps(_mm512_max_ps(v_s, v_lo), v_hi));
            }
        }
        if (t_in && (env_state != t_in)) {
            // [ILP] Branchless AVX-512 add + clamp
            __m512 v_hi = _mm512_set1_ps(10.0f), v_lo = _mm512_set1_ps(-10.0f);
            for (size_t i = 0; i < D; i += 16) {
                __m512 v_s = _mm512_add_ps(_mm512_loadu_ps(env_state + i), _mm512_loadu_ps(t_in + i));
                _mm512_storeu_ps(env_state + i, _mm512_min_ps(_mm512_max_ps(v_s, v_lo), v_hi));
            }
        }

        // ── 2. RECURSIVE THOUGHT CYCLES ──
        nca::layers::glr_step(env_state, weights_.glr_alpha.get(), weights_.glr_beta.get(), env_state, D);
        
        // [HARDENING] Post-Recurrence Clamp (Branchless AVX-512)
        {
            __m512 v_hi = _mm512_set1_ps(10.0f), v_lo = _mm512_set1_ps(-10.0f);
            for (size_t i = 0; i < D; i += 16) {
                __m512 v_s = _mm512_loadu_ps(env_state + i);
                _mm512_storeu_ps(env_state + i, _mm512_min_ps(_mm512_max_ps(v_s, v_lo), v_hi));
            }
        }

        nca::spectral::fwht_inplace({env_state, D});
        
        size_t r_count = 0;
        nca::linalg::mx_quantize_x(env_state, x_q);
        router_->route_to_buffer(x_q, routing_buffer_, &r_count);
        
        if (r_count >= 16) {
            const nca::linalg::MXINT8Tensor* gate_ptrs[16];
            const nca::linalg::MXINT8Tensor* up_ptrs[16];
            for(int j=0; j<16; ++j) {
                size_t id = routing_buffer_[j] % nca::config::N_MICRO_EXPERTS;
                gate_ptrs[j] = &weights_.expert_pool_gate[id];
                up_ptrs[j] = &weights_.expert_pool_up[id];
            }
            alignas(64) float g[16], u[16];
            nca::linalg::mx_rank16_dot_ptrs(gate_ptrs, x_q, g);
            nca::linalg::mx_rank16_dot_ptrs(up_ptrs, x_q, u);

            __m512 vG = _mm512_load_ps(g);
            __m512 vU = _mm512_load_ps(u);
            __m512 vRes = _mm512_mul_ps(nca::simd::avx512::silu_ps(vG), vU);
            _mm512_store_ps(g, vRes);

            {
                __m512 v_hi = _mm512_set1_ps(10.0f), v_lo = _mm512_set1_ps(-10.0f);
                for(int j=0; j<16; ++j) {
                    size_t b_idx = (routing_buffer_[j] * 16) % D;
                    __m512 v_g = _mm512_set1_ps(g[j] * 0.1f);
                    __m512 v_s = _mm512_add_ps(_mm512_loadu_ps(env_state + b_idx), v_g);
                    _mm512_storeu_ps(env_state + b_idx, _mm512_min_ps(_mm512_max_ps(v_s, v_lo), v_hi));
                }
            }
        }
        
        nca::spectral::ifwht_no_scale({env_state, D});

        // ── 3. ACTUATION (Vectorized RMS) ──
        __m512 v_sq0 = _mm512_setzero_ps(), v_sq1 = _mm512_setzero_ps();
        for (size_t j = 0; j < D; j += 32) {
            __m512 v0 = _mm512_loadu_ps(env_state + j);
            __m512 v1 = _mm512_loadu_ps(env_state + j + 16);
            v_sq0 = _mm512_fmadd_ps(v0, v0, v_sq0);
            v_sq1 = _mm512_fmadd_ps(v1, v1, v_sq1);
        }
        float sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(v_sq0, v_sq1)) + 1e-8f;
        float rms = std::sqrt(sum_sq / D);
        float scale = 1.0f / (rms + 1e-7f); // Larger epsilon
        scale = std::clamp(scale, 0.001f, 10.0f); // Tighter scale bounds
        
        size_t write_count = std::min(D, out_dim);
        for(size_t j=0; j<write_count; ++j) e_out[j] = env_state[j] * scale;
    }
}

// ── GEOMETRIC SCHEMA INFERENCE ──────────────────────────────────────────────
// This method completely bypasses the O(N^2) VNNI matrix pipeline.
// It routes the primary wavefront through the compiled Top-16 structural 
// pointers using AVX-512 gather + Xorshift stochastic exploration.
void MultimodalEngine::step_geometric(float temperature) {
    const size_t D = nca::config::D_MODEL;
    float* state = primary_wavefront_->state.get();

    // 1. GLR recurrence (cheap O(N) — preserves the recursive wavefront calculus)
    nca::layers::glr_step(state, weights_.glr_alpha.get(), weights_.glr_beta.get(), state, D);

    // 2. Spectral transform (cheap O(N log N) — preserves frequency-domain reasoning)
    nca::spectral::fwht_inplace({state, D});

    // 3. THE GEOMETRIC HOP — replaces the entire VNNI Expert MoE pipeline
    //    Instead of quantizing, routing through 1024 experts, doing SwiGLU,
    //    we execute a single wavefront step through the compiled pointer graph.
    geometric_router_->step_wavefront(state, temperature);

    // 4. Inverse spectral transform
    nca::spectral::ifwht_no_scale({state, D});
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
        router_->route_to_buffer(x_q, routing_buffer_, &r_count);

        if (r_count >= 16) {
            for(int i=0; i<16; ++i) {
                size_t id = routing_buffer_[i] % nca::config::N_MICRO_EXPERTS;
                nca::linalg::mx_update_gaussian_moment(weights_.expert_pool_gate[id], x_q, adv, 1e-5f, x_norm);
                nca::linalg::mx_update_gaussian_moment(weights_.expert_pool_up[id], x_q, adv, 1e-5f, x_norm);
            }
        }
    }
}

void MultimodalEngine::refine_foundation(const float* state, const float* next_bit_target, float lr_scale) {
    const size_t D = nca::config::D_MODEL;
    nca::linalg::MXUINT8Tensor x_q(D / 32);
    alignas(64) float error_signal[2048];
    for(size_t i=0; i<D; ++i) error_signal[i] = next_bit_target[i] - state[i];
    nca::linalg::mx_quantize_x(state, x_q);
    float x_norm = nca::linalg::mx_compute_activation_norm(x_q);
    size_t r_count = 0;
    router_->route_to_buffer(x_q, routing_buffer_, &r_count);
    if (r_count >= 16) {
        float lr = 1e-6f * lr_scale;
        for(int i=0; i<16; ++i) {
            size_t id = routing_buffer_[i] % nca::config::N_MICRO_EXPERTS;
            float expert_error = error_signal[(id * 16) % D]; 
            nca::linalg::mx_update_gaussian_moment(weights_.expert_pool_gate[id], x_q, expert_error, lr, x_norm);
            nca::linalg::mx_update_gaussian_moment(weights_.expert_pool_up[id], x_q, expert_error, lr, x_norm);
        }
    }
}

void MultimodalEngine::get_saliency_heatmap(float* heatmap_out) {
    const size_t D = nca::config::D_MODEL;
    std::memset(heatmap_out, 0, 256 * sizeof(float));
    for (size_t i = 0; i < D; ++i) heatmap_out[i % 256] += std::abs(primary_wavefront_->state[i]);
    float max_v = 1e-6f;
    for(int i=0; i<256; ++i) max_v = std::max(max_v, heatmap_out[i]);
    for(int i=0; i<256; ++i) heatmap_out[i] /= max_v;
}

WeightRegistry MultimodalEngine::get_weight_registry() {
    WeightRegistry reg;
    const size_t n_experts = nca::config::N_MICRO_EXPERTS;
    reg.expert_pool_gate.reserve(n_experts);
    reg.expert_pool_up.reserve(n_experts);
    for(size_t i=0; i<n_experts; ++i) {
        reg.expert_pool_gate.push_back(&weights_.expert_pool_gate[i]);
        reg.expert_pool_up.push_back(&weights_.expert_pool_up[i]);
    }
    reg.halting_gate = &weights_.halting_gate;
    reg.vision_A = weights_.vision_A.get();
    reg.vision_B = weights_.vision_B.get();
    reg.vision_C = weights_.vision_C.get();
    reg.glr_alpha = weights_.glr_alpha.get();
    reg.glr_beta = weights_.glr_beta.get();
    reg.d_model = nca::config::D_MODEL;
    reg.n_experts = nca::config::N_MICRO_EXPERTS;
    return reg;
}

} // namespace nca::execution
