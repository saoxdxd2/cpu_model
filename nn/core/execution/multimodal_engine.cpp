// ============================================================================
// NCA — Multimodal Engine Implementation
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
#include <cmath>
#include <algorithm>
#include <span>
#include <iostream>

namespace nca::execution {

MultimodalEngine::MultimodalEngine() {
    state_ = nca::simd::make_aligned_unique<float[]>(nca::config::D_MODEL);
    vision_latent_ = nca::simd::make_aligned_unique<float[]>(nca::config::D_MODEL);
    h_glr_ = nca::simd::make_aligned_unique<float[]>(nca::config::D_MODEL);
    h_ssm_ = nca::simd::make_aligned_unique<float[]>(nca::config::D_MODEL);
    
    for(size_t i=0; i<nca::config::D_MODEL; ++i) {
        state_[i] = 0.0f; h_glr_[i] = 0.0f; h_ssm_[i] = 0.0f;
    }
}

void MultimodalEngine::step(const float* text_in, const float* image_in, float* out) {
    if (image_in) {
        nca::vision::ScannerConfig scan_cfg; // default 16, 16, 128
        size_t n_patches = scan_cfg.H * scan_cfg.W;
        size_t d_state = 16;
        
        auto A = nca::simd::make_aligned_unique<float[]>(n_patches * d_state);
        auto B = nca::simd::make_aligned_unique<float[]>(d_state);
        auto C = nca::simd::make_aligned_unique<float[]>(d_state);
        auto h = nca::simd::make_aligned_unique<float[]>(n_patches * d_state);
        auto y = nca::simd::make_aligned_unique<float[]>(n_patches * scan_cfg.C);

        nca::vision::ssm2d_scan(h.get(), A.get(), B.get(), C.get(), image_in, y.get(), scan_cfg);
        
        nca::vision::SpectralPruner::Config pruner_cfg = { n_patches, scan_cfg.C, 0.1f };
        nca::vision::SpectralPruner pruner(pruner_cfg);
        std::vector<size_t> active_indices(n_patches);
        pruner.prune(std::span<const float>(y.get(), n_patches * scan_cfg.C), active_indices);
        
        LatentAdapter adapter;
        adapter.project(y.get(), vision_latent_.get());
        
        for(size_t i=0; i<nca::config::D_MODEL; ++i) state_[i] += vision_latent_[i];
    }

    float accumulated_h = 0.0f;
    int cycles = 0;
    nca::layers::HaltingState h_state;

    while (accumulated_h < nca::config::ACT_HALT_THRESHOLD && cycles < nca::config::MAX_ACT_CYCLES) {
        auto a = nca::simd::make_aligned_unique<float[]>(nca::config::D_MODEL);
        auto b = nca::simd::make_aligned_unique<float[]>(nca::config::D_MODEL);
        std::fill(a.get(), a.get() + nca::config::D_MODEL, 0.5f);
        std::fill(b.get(), b.get() + nca::config::D_MODEL, 0.5f);

        nca::layers::glr_step(h_glr_.get(), a.get(), b.get(), state_.get(), nca::config::D_MODEL);
        for(size_t i=0; i<nca::config::D_MODEL; ++i) state_[i] += h_glr_[i];
        
        nca::linalg::MXUINT8Tensor x_q(nca::config::D_MODEL);
        nca::linalg::mx_quantize_x(state_.get(), x_q);
        
        nca::linalg::MXINT8Tensor w_h(nca::config::D_MODEL);
        auto wh_f = nca::simd::make_aligned_unique<float[]>(nca::config::D_MODEL);
        std::fill(wh_f.get(), wh_f.get() + nca::config::D_MODEL, 0.01f);
        nca::linalg::mx_quantize_w(wh_f.get(), w_h);

        bool should_halt = false;
        accumulated_h += nca::layers::halting_step(x_q, w_h, 0.0f, h_state, should_halt);
        if(should_halt) break;
        cycles++;
    }

    if (out) std::copy(state_.get(), state_.get() + nca::config::D_MODEL, out);
}

} // namespace nca::execution
