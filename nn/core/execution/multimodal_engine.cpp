// ============================================================================
// NCA — Fused Multimodal Inference Engine
// core/execution/multimodal_engine.cpp
// ============================================================================

#include "core/execution/multimodal_engine.hpp"
#include "core/execution/route_planner.hpp"
#include "core/simd/dispatch.hpp"

namespace nca::execution {

MultimodalEngine::MultimodalEngine(Config cfg) 
    : cfg_(cfg),
      pruner_(std::make_unique<nca::vision::SpectralPruner>(cfg.prune_cfg)),
      adapter_(std::make_unique<nca::execution::LatentAdapter>(cfg.adapter_cfg)) {
    
    const size_t d_inner = cfg.vision_cfg.H * cfg.vision_cfg.W * cfg.vision_cfg.C;
    scanner_buf_.resize(d_inner);
    active_indices_.resize(cfg.prune_cfg.n_tokens);
    shuffled_buf_.resize(cfg.prune_cfg.n_tokens * cfg.prune_cfg.d_model);
}

void MultimodalEngine::step(
    const float* __restrict image_in,
    float* __restrict hidden_state,
    float* __restrict logic_output
) {
    // ── STAGE 1: VISION SCAN & PRUNE ────────────────────────────────────────
    // (Note: weights passed here are examples for the architectural flow)
    size_t k = pruner_->prune(scanner_buf_, active_indices_);

    // ── STAGE 2: DATA LOCALITY SHUFFLE ──────────────────────────────────────
    nca::execution::RoutePlan plan(cfg_.prune_cfg.n_tokens);
    plan.num_active = k;
    std::copy(active_indices_.begin(), active_indices_.begin() + k, plan.active_indices);
    nca::execution::shuffle_active_tokens(scanner_buf_.data(), shuffled_buf_.data(), plan, cfg_.prune_cfg.d_model);

    // ── STAGE 3: LATENT PROJECTION (Bridge) ─────────────────────────────────
    nca::linalg::MXUINT8Tensor x_q(shuffled_buf_.size() / 32);
    nca::linalg::mx_quantize_x(shuffled_buf_.data(), x_q);
    adapter_->project(x_q, hidden_state);

    // ── STAGE 4: TRIPLE-HYBRID REASONING (Logic) ────────────────────────────
    // This is where the actual "Neural Network" reasoning happens.
    // We execute the recurrent backbone with the newly projected visual context.
    
    // 1. GLR Baseline (65% path)
    nca::layers::glr_step(hidden_state, nullptr, nullptr, hidden_state, cfg_.d_model);

    // 2. Selective SSM Dynamics (20% path)
    nca::layers::SSMConfig ssm_cfg{cfg_.d_model, 16};
    nca::layers::ssm_step(nullptr, nullptr, nullptr, nullptr, hidden_state, logic_output, ssm_cfg);

    // 3. ACT Halting Gate (System 2 Thinking)
    bool should_halt = false;
    nca::layers::halting_step(x_q, nca::linalg::MXINT8Tensor(0), 0.0f, halt_state_, should_halt);
}

} // namespace nca::execution
