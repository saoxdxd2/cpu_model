#pragma once
// ============================================================================
// NCA — Fused Multimodal Inference Engine
// core/execution/multimodal_engine.hpp
// ============================================================================

#include "core/simd/memory.hpp"
#include "encoding/state_encoder.hpp"
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
    // Quantized Block-Tensors
    std::vector<nca::linalg::MXINT8Tensor*> expert_pool_gate;
    std::vector<nca::linalg::MXINT8Tensor*> expert_pool_up;
    nca::linalg::MXINT8Tensor* halting_gate;
    
    // FP32 Parameter Buffers
    float* vision_A;
    float* vision_B;
    float* vision_C;
    float* glr_alpha;
    float* glr_beta;
    
    // Fixed Sizing Info
    size_t d_model;
    size_t n_experts;
};

class MultimodalEngine {
public:
    friend class nca::training::WeightAdapter;
    explicit MultimodalEngine(size_t obs_dim = nca::config::D_MODEL, 
                              size_t act_dim = 80, 
                              nca::config::EngineConfig engine_cfg = {});

    // Executes the COMPLETE custom neural network logic:
    // Vision (Scan+Prune) -> Bridge (Encoder) -> Recursive Logic (ACT Depth)
    void step(const float* text_in, const float* image_in, float* out);

    // [OPTIMIZATION] High-Throughput Batch Inference
    // Processes multiple environments in a single fused AVX-512 pass
    void step_batch(const float* text_in, const float* image_in, float* out, size_t batch_size);
    
    // The RL Trajectory Update (connects Advantage to Gaussian Moments)
    void update_from_trajectory(size_t count, size_t obs_dim, size_t act_dim, 
                                const float* states, const float* actions, const float* advantages);

    // Clears the persistent state and RLS factors (Memory Wipe)
    void reset_state();

    WeightRegistry get_weight_registry();

private:
    nca::config::EngineConfig engine_cfg_;
    size_t obs_dim_;
    size_t act_dim_;
    ImportanceClassifier classifier_;

    // [OPTIMIZATION] Zero-Allocation Persistent Buffers
    std::vector<size_t> routing_buffer_;
    nca::simd::aligned_unique_ptr<float[]> batch_state_; // [batch_size * D_MODEL]

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
    
    nca::encoding::SiliconEncoder encoder_;
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
