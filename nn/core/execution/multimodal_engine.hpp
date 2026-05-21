#pragma once
// ============================================================================
// NCA — Fused Multimodal Inference Engine
// core/execution/multimodal_engine.hpp
// ============================================================================

#include "core/simd/memory.hpp"
#include "core/execution/latent_adapter.hpp"
#include "core/execution/importance.hpp"
#include "core/linalg/mx_linear.hpp"
#include "core/linalg/hashed_router.hpp"
#include "core/spectral/kronecker_rls.hpp"
#include "config/model_config.hpp"
#include <memory>
#include <vector>

namespace nca::execution {

class MultimodalEngine {
public:
    explicit MultimodalEngine(nca::config::EngineConfig engine_cfg = {});

    // Executes the COMPLETE custom neural network logic:
    // Vision (Scan+Prune) -> Bridge (Adapter) -> Recursive Logic (ACT Depth)
    void step(const float* text_in, const float* image_in, float* out);

    // Clears the persistent state and RLS factors (Memory Wipe)
    void reset_state();

private:
    nca::config::EngineConfig engine_cfg_;
    ImportanceClassifier classifier_;

    // ── PERSISTENT WEIGHT ANCHORS ───────────────────────────────────────────
    nca::simd::aligned_unique_ptr<float[]> W_vision_A_;
    nca::simd::aligned_unique_ptr<float[]> W_vision_B_;
    nca::simd::aligned_unique_ptr<float[]> W_vision_C_;
    nca::simd::aligned_unique_ptr<float[]> W_glr_alpha_;
    nca::simd::aligned_unique_ptr<float[]> W_glr_beta_;
    
    // SDMS Expert Pool
    std::vector<nca::linalg::MXINT8Tensor> expert_pool_gate_;
    std::vector<nca::linalg::MXINT8Tensor> expert_pool_up_;
    std::vector<nca::linalg::MXINT8Tensor> shared_experts_gate_;
    std::vector<nca::linalg::MXINT8Tensor> shared_experts_up_;
    
    std::unique_ptr<nca::linalg::HashedRouter> router_;
    
    nca::execution::LatentAdapter adapter_;
    nca::linalg::MXINT8Tensor W_halt_;

    // ── SPECTRAL LONG-TERM MEMORY (Titans-style) ────────────────────────────
    std::unique_ptr<nca::spectral::KroneckerRLSState> spectral_rls_;

    // ── PERSISTENT STATE BUFFERS (L1-HOT) ───────────────────────────────────
    nca::simd::aligned_unique_ptr<float[]> state_;
    nca::simd::aligned_unique_ptr<float[]> vision_latent_;
    nca::simd::aligned_unique_ptr<float[]> h_glr_;
    nca::simd::aligned_unique_ptr<float[]> state_momentum_;
    nca::simd::aligned_unique_ptr<float[]> prediction_buf_;
};

} // namespace nca::execution
