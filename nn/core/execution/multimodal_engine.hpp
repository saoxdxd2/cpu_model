#pragma once
// ============================================================================
// NCA — Saturated Multimodal Inference Engine (v29.0 - Hardened)
// core/execution/multimodal_engine.hpp
// ============================================================================

#include "core/simd/memory.hpp"
#include "encoding/state_encoder.hpp"
#include "encoding/vision_encoder.hpp"
#include "core/execution/importance.hpp"
#include "core/linalg/mx_linear.hpp"
#include "core/linalg/hashed_router.hpp"
#include "core/spectral/kronecker_rls.hpp"
#include "config/model_config.hpp"
#include <memory>
#include <vector>

namespace nca::training { class WeightAdapter; }

namespace nca::execution {

struct WeightRegistry {
    std::vector<nca::linalg::MXINT8Tensor*> expert_pool_gate;
    std::vector<nca::linalg::MXINT8Tensor*> expert_pool_up;
    nca::linalg::MXINT8Tensor* halting_gate;
    float* vision_A;
    float* vision_B;
    float* vision_C;
    float* glr_alpha;
    float* glr_beta;
    size_t d_model;
    size_t n_experts;
};

class MultimodalEngine {
public:
    friend class nca::training::WeightAdapter;
    explicit MultimodalEngine(size_t obs_dim = nca::config::D_MODEL, 
                              size_t act_dim = 80, 
                              nca::config::EngineConfig engine_cfg = {});

    // Primary High-Throughput Batch Inference
    void step_batch(const float* text_in, const float* image_in, float* out, size_t batch_size);

    // Legacy Single-Step (Wraps step_batch)
    void step(const float* text_in, const float* image_in, float* out) {
        step_batch(text_in, image_in, out, 1);
    }
    
    // GAE-Driven Online Adaptation
    void update_from_trajectory(size_t count, size_t obs_dim, size_t act_dim, 
                                const float* states, const float* actions, const float* advantages);

    void reset_state();
    WeightRegistry get_weight_registry();

private:
    nca::config::EngineConfig engine_cfg_;
    size_t obs_dim_;
    size_t act_dim_;
    ImportanceClassifier classifier_;
    
    // [HARDENED] Silicon-Aligned Buffers (Zero Allocations)
    alignas(64) size_t routing_buffer_[1024]; 
    nca::simd::aligned_unique_ptr<float[]> batch_state_; 

    // ── PERSISTENT WEIGHT ANCHORS ───────────────────────────────────────────
    nca::simd::aligned_unique_ptr<float[]> W_vision_A_;
    nca::simd::aligned_unique_ptr<float[]> W_vision_B_;
    nca::simd::aligned_unique_ptr<float[]> W_vision_C_;
    nca::simd::aligned_unique_ptr<float[]> W_glr_alpha_;
    nca::simd::aligned_unique_ptr<float[]> W_glr_beta_;
    
    std::vector<nca::linalg::MXINT8Tensor> expert_pool_gate_;
    std::vector<nca::linalg::MXINT8Tensor> expert_pool_up_;
    
    std::unique_ptr<nca::linalg::HashedRouter> router_;
    nca::encoding::SiliconEncoder encoder_;
    nca::encoding::SiliconVisionEncoder vision_encoder_;
    nca::linalg::MXINT8Tensor W_halt_;

    std::unique_ptr<nca::spectral::KroneckerRLSState> spectral_rls_;

    // ── PERSISTENT STATE BUFFERS (L1-HOT) ───────────────────────────────────
    nca::simd::aligned_unique_ptr<float[]> state_;
    nca::simd::aligned_unique_ptr<float[]> vision_latent_;
    nca::simd::aligned_unique_ptr<float[]> h_glr_;
    nca::simd::aligned_unique_ptr<float[]> state_momentum_;
    nca::simd::aligned_unique_ptr<float[]> prediction_buf_;
};

} // namespace nca::execution
