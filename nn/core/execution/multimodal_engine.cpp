// ============================================================================
// NCA — Multimodal Engine Implementation (Refined Parameter Contract)
// core/execution/multimodal_engine.cpp
// ============================================================================

#include "core/execution/multimodal_engine.hpp"
#include "config/model_config.hpp"
#include "core/simd/memory.hpp"
#include "core/vision/scanner.hpp"
#include "core/vision/spectral_pruner.hpp"
#include "core/execution/latent_adapter.hpp"
#include "core/layers/glr.hpp"
#include "core/layers/ssm.hpp"
#include "core/layers/sla.hpp"
#include "core/layers/mlp.hpp"
#include "core/layers/halting.hpp"
#include "core/activations.hpp"
#include <cmath>
#include <algorithm>
#include <span>
#include <random>

namespace nca::execution {

MultimodalEngine::MultimodalEngine() {
    const size_t D = nca::config::D_MODEL;
    const size_t di_vision = 16 * 16 * 128; 
    const size_t d_state = 16;
    const size_t n_experts = 256; 

    // ── STABLE PARAMETER CONTRACT: FIXED SEED INITIALIZATION ───────────────
    // This ensures that the model is deterministic and verifiable until
    // we implement the real .safetensors loader in Phase 17.
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    auto init_f = [&](float* p, size_t n) { for(size_t i=0; i<n; ++i) p[i] = dist(gen); };

    // 1. Initialize Vision Anchors
    W_vision_A_ = nca::simd::make_aligned_unique<float[]>(di_vision * d_state);
    W_vision_B_ = nca::simd::make_aligned_unique<float[]>(d_state);
    W_vision_C_ = nca::simd::make_aligned_unique<float[]>(d_state);
    init_f(W_vision_A_.get(), di_vision * d_state);
    init_f(W_vision_B_.get(), d_state);
    init_f(W_vision_C_.get(), d_state);

    // 2. Initialize Logic Anchors (Backbone)
    W_glr_alpha_ = nca::simd::make_aligned_unique<float[]>(D);
    W_glr_beta_ = nca::simd::make_aligned_unique<float[]>(D);
    init_f(W_glr_alpha_.get(), D);
    init_f(W_glr_beta_.get(), D);

    // 3. Initialize Hashed Router (Saturated VNNI)
    nca::linalg::HashedRouter::Config router_cfg = { D, n_experts, 4, 16 };
    router_ = std::make_unique<nca::linalg::HashedRouter>(router_cfg);

    // 4. Initialize Expert Pool (Associative Memory)
    W_mlp_gate_pool_.reserve(n_experts * 16);
    W_mlp_up_pool_.reserve(n_experts * 16);
    auto temp = nca::simd::make_aligned_unique<float[]>(D);
    for(size_t i=0; i < n_experts * 16; ++i) {
        W_mlp_gate_pool_.emplace_back(D / 32);
        W_mlp_up_pool_.emplace_back(D / 32);
        init_f(temp.get(), D); nca::linalg::mx_quantize_w(temp.get(), W_mlp_gate_pool_[i]);
        init_f(temp.get(), D); nca::linalg::mx_quantize_w(temp.get(), W_mlp_up_pool_[i]);
    }

    // 5. Initialize Halting Anchor
    W_halt_ = nca::linalg::MXINT8Tensor(D / 32);
    init_f(temp.get(), D); nca::linalg::mx_quantize_w(temp.get(), W_halt_);

    // ── ALLOCATE L1-HOT STATE BUFFERS ───────────────────────────────────────
    state_ = nca::simd::make_aligned_unique<float[]>(D);
    vision_latent_ = nca::simd::make_aligned_unique<float[]>(D);
    h_glr_ = nca::simd::make_aligned_unique<float[]>(D);
    
    std::fill(state_.get(), state_.get() + D, 0.0f);
}

void MultimodalEngine::step(const float* text_in, const float* image_in, float* out) {
    const size_t D = nca::config::D_MODEL;
    if (text_in) std::copy(text_in, text_in + D, state_.get());

    if (image_in) {
        nca::vision::ScannerConfig scan_cfg; 
        size_t n_patches = scan_cfg.H * scan_cfg.W;
        size_t di = n_patches * scan_cfg.C;
        auto y_scan = nca::simd::make_aligned_unique<float[]>(di);
        auto h_scan = nca::simd::make_aligned_unique<float[]>(di * 16);

        nca::vision::ssm2d_scan(h_scan.get(), W_vision_A_.get(), W_vision_B_.get(), W_vision_C_.get(), image_in, y_scan.get(), scan_cfg);
        
        nca::vision::SpectralPruner::Config pruner_cfg = { n_patches, scan_cfg.C, 0.25f };
        nca::vision::SpectralPruner pruner(pruner_cfg);
        std::vector<size_t> active_indices(n_patches);
        size_t kept = pruner.prune(std::span<const float>(y_scan.get(), di), active_indices);
        
        std::vector<bool> mask(n_patches, false);
        for(size_t i=0; i<kept; ++i) mask[active_indices[i]] = true;
        for(size_t i=0; i<n_patches; ++i) if(!mask[i]) std::fill(y_scan.get() + i*scan_cfg.C, y_scan.get() + (i+1)*scan_cfg.C, 0.0f);

        alignas(64) float adapter_out[2048]; 
        adapter_.project(y_scan.get(), adapter_out);
        for(size_t i=0; i<D; ++i) state_[i] += adapter_out[i];
    }

    float accumulated_h = 0.0f;
    int cycles = 0;
    nca::layers::HaltingState h_state;
    nca::linalg::MXUINT8Tensor x_q(D / 32);
    std::vector<size_t> active_logic_blocks;

    while (accumulated_h < nca::config::ACT_HALT_THRESHOLD && cycles < nca::config::MAX_ACT_CYCLES) {
        nca::layers::glr_step(h_glr_.get(), W_glr_alpha_.get(), W_glr_beta_.get(), state_.get(), D);
        for(size_t i=0; i<D; ++i) state_[i] += h_glr_[i];

        router_->route(state_.get(), active_logic_blocks);
        nca::linalg::mx_quantize_x(state_.get(), x_q);

        for (size_t block_idx : active_logic_blocks) {
            alignas(64) float gate_res[16], up_res[16];
            nca::linalg::mx_rank16_dot(&W_mlp_gate_pool_[block_idx * 16], x_q, gate_res);
            nca::linalg::mx_rank16_dot(&W_mlp_up_pool_[block_idx * 16], x_q, up_res);
            for(int i=0; i<16; ++i) {
                float silu = gate_res[i] / (1.0f + std::exp(-gate_res[i]));
                state_[(block_idx * 16 + i) % D] += silu * up_res[i];
            }
        }

        bool should_halt = false;
        accumulated_h += nca::layers::halting_step(x_q, W_halt_, 0.0f, h_state, should_halt);
        if(should_halt) break;
        cycles++;
    }

    if (out) std::copy(state_.get(), state_.get() + D, out);
}

} // namespace nca::execution
