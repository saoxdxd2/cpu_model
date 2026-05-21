// ============================================================================
// NCA — Multimodal Engine Implementation (Dual-Backend v7.0)
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
#include <cmath>
#include <algorithm>
#include <span>
#include <random>

namespace nca::execution {

MultimodalEngine::MultimodalEngine(nca::config::EngineConfig engine_cfg) 
    : engine_cfg_(engine_cfg) {
    const size_t D = nca::config::D_MODEL;
    const size_t n_experts = 256; 

    // 1. INITIALIZE WEIGHTS (Deterministic Contract)
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(-0.01f, 0.01f);
    auto init_f = [&](float* p, size_t n) { for(size_t i=0; i<n; ++i) p[i] = dist(gen); };

    W_vision_A_ = nca::simd::make_aligned_unique<float[]>(16 * 16 * 128 * 16);
    W_vision_B_ = nca::simd::make_aligned_unique<float[]>(16);
    W_vision_C_ = nca::simd::make_aligned_unique<float[]>(16);
    init_f(W_vision_A_.get(), 16 * 16 * 128 * 16);
    init_f(W_vision_B_.get(), 16);
    init_f(W_vision_C_.get(), 16);

    W_glr_alpha_ = nca::simd::make_aligned_unique<float[]>(D);
    W_glr_beta_ = nca::simd::make_aligned_unique<float[]>(D);
    init_f(W_glr_alpha_.get(), D);
    init_f(W_glr_beta_.get(), D);

    router_ = std::make_unique<nca::linalg::HashedRouter>(nca::linalg::HashedRouter::Config{D, n_experts, 4, 16});

    W_mlp_gate_pool_.reserve(n_experts * 16);
    W_mlp_up_pool_.reserve(n_experts * 16);
    auto temp = nca::simd::make_aligned_unique<float[]>(D);
    for(size_t i=0; i < n_experts * 16; ++i) {
        W_mlp_gate_pool_.emplace_back(D / 32);
        W_mlp_up_pool_.emplace_back(D / 32);
        init_f(temp.get(), D); nca::linalg::mx_quantize_w(temp.get(), W_mlp_gate_pool_[i]);
        init_f(temp.get(), D); nca::linalg::mx_quantize_w(temp.get(), W_mlp_up_pool_[i]);
    }

    W_halt_ = nca::linalg::MXINT8Tensor(D / 32);
    init_f(temp.get(), D); nca::linalg::mx_quantize_w(temp.get(), W_halt_);

    // 2. INITIALIZE SPECTRAL STATE
    spectral_rls_ = std::make_unique<nca::spectral::KroneckerRLSState>(D);

    // 3. ALLOCATE STATE BUFFERS
    state_ = nca::simd::make_aligned_unique<float[]>(D);
    vision_latent_ = nca::simd::make_aligned_unique<float[]>(D);
    h_glr_ = nca::simd::make_aligned_unique<float[]>(D);
    std::fill(state_.get(), state_.get() + D, 0.0f);
}

void MultimodalEngine::step(const float* text_in, const float* image_in, float* out) {
    const size_t D = nca::config::D_MODEL;
    if (text_in) std::copy(text_in, text_in + D, state_.get());

    // Reset SRLS state for deterministic per-step solving
    if (engine_cfg_.logic_backend == nca::config::LogicBackend::SpectralRLS) {
        spectral_rls_->reset();
    }

    if (image_in) {
        nca::vision::ScannerConfig scan_cfg; 
        size_t di = scan_cfg.H * scan_cfg.W * scan_cfg.C; 
        auto y_scan = nca::simd::make_aligned_unique<float[]>(di);
        auto h_scan = nca::simd::make_aligned_unique<float[]>(di * 16);

        nca::vision::ssm2d_scan(h_scan.get(), W_vision_A_.get(), W_vision_B_.get(), W_vision_C_.get(), image_in, y_scan.get(), scan_cfg);
        
        std::vector<size_t> active_indices(scan_cfg.H * scan_cfg.W);
        nca::vision::SpectralPruner pruner({scan_cfg.H * scan_cfg.W, scan_cfg.C, 0.25f});
        size_t kept = pruner.prune(std::span<const float>(y_scan.get(), di), active_indices);
        
        std::vector<bool> mask(scan_cfg.H * scan_cfg.W, false);
        for(size_t i=0; i<kept; ++i) mask[active_indices[i]] = true;
        for(size_t i=0; i<scan_cfg.H * scan_cfg.W; ++i) if(!mask[i]) std::fill(y_scan.get() + i*scan_cfg.C, y_scan.get() + (i+1)*scan_cfg.C, 0.0f);

        alignas(64) float adapter_out[2048]; 
        adapter_.project(y_scan.get(), adapter_out);
        for(size_t i=0; i<D; ++i) state_[i] += adapter_out[i];
    }

    // ── RECURSIVE ACT LOOP ──────────────────────────────────────────────────
    float accumulated_h = 0.0f;
    int cycles = 0;
    nca::layers::HaltingState h_state;
    nca::linalg::MXUINT8Tensor x_q(D / 32);

    while (accumulated_h < nca::config::ACT_HALT_THRESHOLD && cycles < nca::config::MAX_ACT_CYCLES) {
        // 1. Temporal Proposal (Always active as teacher)
        alignas(64) float glr_prop[2048];
        nca::layers::glr_step(glr_prop, W_glr_alpha_.get(), W_glr_beta_.get(), state_.get(), D);

        if (engine_cfg_.logic_backend == nca::config::LogicBackend::SpectralRLS) {
            // BACKEND A: Spectral RLS (The "Perfect Solver")
            spectral_logic_step(state_.get(), glr_prop, *spectral_rls_, D);
        } else {
            // BACKEND B: Hashed Expert Pool (The "Dynamic Router")
            for(size_t i=0; i<D; ++i) state_[i] += glr_prop[i];
            
            std::vector<size_t> active_logic_blocks;
            router_->route(state_.get(), active_logic_blocks);
            nca::linalg::mx_quantize_x(state_.get(), x_q);

            for (size_t block_idx : active_logic_blocks) {
                alignas(64) float g[16], u[16];
                nca::linalg::mx_rank16_dot(&W_mlp_gate_pool_[block_idx * 16], x_q, g);
                nca::linalg::mx_rank16_dot(&W_mlp_up_pool_[block_idx * 16], x_q, u);
                for(int i=0; i<16; ++i) state_[(block_idx * 16 + i) % D] += (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
            }
        }

        // 2. Halting Check
        nca::linalg::mx_quantize_x(state_.get(), x_q);
        bool halt = false;
        accumulated_h += nca::layers::halting_step(x_q, W_halt_, 0.0f, h_state, halt);
        if(halt) break;
        cycles++;
    }
    if (out) std::copy(state_.get(), state_.get() + D, out);
}

} // namespace nca::execution
