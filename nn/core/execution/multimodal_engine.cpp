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
    weights_.initialize_binary_curve(D, D); // Pre-compile weights into binary masks
    primary_wavefront_ = std::make_unique<SiliconWavefront>(D);
    
    spectral_rls_ = std::make_unique<nca::spectral::KroneckerRLSState>(D);

    // [GEOMETRIC] Initialize the Wavefront Router (dormant until a .geo graph is loaded)
    geometric_router_ = std::make_unique<WavefrontRouter>(D, 16);
    geometric_router_->initialize_default_graph();
}

void MultimodalEngine::reset_state() {
    primary_wavefront_->reset(nca::config::D_MODEL);
    spectral_rls_->reset();
}



void MultimodalEngine::step_swarm(const float* initial_input, float* swarm_out, size_t max_agents) {
    const size_t D = nca::config::D_MODEL;
    const size_t out_dim = act_dim_ + 1;
    
    if (max_agents > 128) max_agents = 128;

    float* current_context = const_cast<float*>(initial_input);
    float accumulated_h = 0.0f;
    size_t active_swarm_size = 0;

    for (size_t n = 0; n < max_agents; ++n) {
        float* agent_state = primary_wavefront_->state.get();
        float* agent_out = swarm_out + n * out_dim;
        
        // ── 1. CHAINED INGESTION ──
        std::memset(agent_state, 0, D * sizeof(float));
        for (size_t i = 0; i < D; ++i) agent_state[i] += current_context[i];

        // ── 2. RECURSIVE THOUGHT CYCLE (Geometric Schema) ──
        step_geometric(nullptr, nullptr, agent_out, 0.2f);

        // ── 3. ADAPTIVE HALTING (Silicon-ACT) ──
        float h = 1.0f / (1.0f + std::exp(-agent_out[0]));
        accumulated_h += h;
        active_swarm_size++;

        if (accumulated_h > nca::config::ACT_HALT_THRESHOLD) break;
        current_context = agent_state; 
    }

    std::cout << "  [SWARM] Wavefront reached consensus with " << active_swarm_size << " agents.\n";
}



void MultimodalEngine::step_geometric_env(const float* obs_in, float* out, float temperature) {
    const size_t D = nca::config::D_MODEL;
    alignas(64) float latent[2048];
    encoder_.encode(obs_in, latent, D);
    step_geometric(latent, nullptr, out, temperature);
}

// ── GEOMETRIC SCHEMA INFERENCE ──────────────────────────────────────────────
// This method completely bypasses the O(N^2) VNNI matrix pipeline.
// It routes the primary wavefront through the compiled Top-16 structural 
// pointers using AVX-512 gather + Xorshift stochastic exploration.
void MultimodalEngine::step_geometric(const float* text_in, const float* image_in, float* out, float temperature) {
    const size_t D = nca::config::D_MODEL;
    const size_t out_dim = act_dim_ + 1;
    
    // We operate entirely on the primary wavefront.
    // In Swarm mode, step_swarm copies the context into primary_wavefront_ before calling this.
    float* env_state = primary_wavefront_->state.get();

    // ── 0. INGESTION (Text & Vision) ──
    if (text_in && text_in != env_state) {
        __m512 v_hi = _mm512_set1_ps(10.0f), v_lo = _mm512_set1_ps(-10.0f);
        for (size_t i = 0; i < D; i += 16) {
            __m512 v_s = _mm512_add_ps(_mm512_loadu_ps(env_state + i), _mm512_loadu_ps(text_in + i));
            _mm512_storeu_ps(env_state + i, _mm512_min_ps(_mm512_max_ps(v_s, v_lo), v_hi));
        }
    }
    
    if (image_in) {
        vision_encoder_.encode_gui(image_in, primary_wavefront_->prediction_buf.get(), D);
        float gate = text_in ? (1.0f + std::tanh(text_in[0])) : 0.1f;
        __m512 v_gate = _mm512_set1_ps(gate);
        __m512 v_hi = _mm512_set1_ps(10.0f), v_lo = _mm512_set1_ps(-10.0f);
        for (size_t i = 0; i < D; i += 16) {
            __m512 v_s = _mm512_loadu_ps(env_state + i);
            __m512 v_p = _mm512_loadu_ps(primary_wavefront_->prediction_buf.get() + i);
            v_s = _mm512_fmadd_ps(v_p, v_gate, v_s);
            _mm512_storeu_ps(env_state + i, _mm512_min_ps(_mm512_max_ps(v_s, v_lo), v_hi));
        }
    }

    // ── 1. GLR RECURRENCE ──
    nca::layers::glr_step(env_state, weights_.glr_alpha.get(), weights_.glr_beta.get(), env_state, D);

    // ── 2. SPECTRAL DOMAIN ──
    nca::spectral::fwht_inplace({env_state, D});

    // ── 3. GEOMETRIC HOP (Top-16 SIMD Wavefront) ──
    geometric_router_->step_wavefront(env_state, temperature);

    // ── 3.5 BINARY CURVE TREE (Transistor-Level Weight Routing) ──
    // The weights are not multipliers, they are binary logic gates.
    alignas(64) float bct_out[2048] = {0.0f}; // Assumes D_MODEL <= 2048
    size_t num_masks = weights_.bct_n / 16;
    for (size_t i = 0; i < weights_.bct_m; ++i) {
        __m512 v_sum = _mm512_setzero_ps();
        const uint16_t* row_masks = &weights_.binary_curve_masks[i * num_masks];
        for (size_t j = 0; j < weights_.bct_n; j += 16) {
            __mmask16 gate_mask = row_masks[j / 16];
            __m512 v_routed_signal = _mm512_maskz_loadu_ps(gate_mask, env_state + j);
            v_sum = _mm512_add_ps(v_sum, v_routed_signal);
        }
        bct_out[i] = _mm512_reduce_add_ps(v_sum) * weights_.bct_scale + weights_.bct_offset;
    }
    std::memcpy(env_state, bct_out, D * sizeof(float));

    // ── 4. INVERSE SPECTRAL ──
    nca::spectral::ifwht_no_scale({env_state, D});
    
    // ── 5. ACTUATION (RMS Scale) ──
    if (out) {
        __m512 v_sq0 = _mm512_setzero_ps(), v_sq1 = _mm512_setzero_ps();
        for (size_t j = 0; j < D; j += 32) {
            __m512 v0 = _mm512_loadu_ps(env_state + j);
            __m512 v1 = _mm512_loadu_ps(env_state + j + 16);
            v_sq0 = _mm512_fmadd_ps(v0, v0, v_sq0);
            v_sq1 = _mm512_fmadd_ps(v1, v1, v_sq1);
        }
        float sum_sq = _mm512_reduce_add_ps(_mm512_add_ps(v_sq0, v_sq1)) + 1e-8f;
        float rms = std::sqrt(sum_sq / D);
        float scale = 1.0f / (rms + 1e-7f);
        scale = std::clamp(scale, 0.001f, 10.0f);
        
        size_t write_count = std::min(D, out_dim);
        for(size_t j = 0; j < write_count; ++j) out[j] = env_state[j] * scale;
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
    reg.vision_A = weights_.vision_A.get();
    reg.vision_B = weights_.vision_B.get();
    reg.vision_C = weights_.vision_C.get();
    reg.glr_alpha = weights_.glr_alpha.get();
    reg.glr_beta = weights_.glr_beta.get();
    reg.d_model = nca::config::D_MODEL;
    return reg;
}

} // namespace nca::execution
