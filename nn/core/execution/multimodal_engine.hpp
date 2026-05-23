#pragma once
// ============================================================================
// NCA — Saturated Multimodal Inference Engine (v31.0 - Hardened)
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
#include "nn/core/execution/silicon_memory.hpp"
#include <memory>
#include <vector>

namespace nca::training { class WeightAdapter; }

namespace nca::execution {

struct WeightRegistry {
    std::vector<::nca::linalg::MXINT8Tensor*> expert_pool_gate;
    std::vector<::nca::linalg::MXINT8Tensor*> expert_pool_up;
    ::nca::linalg::MXINT8Tensor* halting_gate;
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
    friend class ::nca::training::WeightAdapter;
    explicit MultimodalEngine(size_t obs_dim = ::nca::config::D_MODEL, 
                              size_t act_dim = 80, 
                              ::nca::config::EngineConfig engine_cfg = {});

    // Primary High-Throughput Batch Inference
    void step_batch(const float* text_in, const float* image_in, float* out, size_t batch_size);

    // [NEW] Silicon Swarm: Chained Recurrence
    // Output of agent N becomes the context for agent N+1
    void step_swarm(const float* initial_input, float* swarm_out, size_t max_agents);

    // Legacy Single-Step
    void step(const float* text_in, const float* image_in, float* out);
    
    // GAE-Driven Online Adaptation (Fast Context)
    void update_from_trajectory(size_t count, size_t obs_dim, size_t act_dim, 
                                const float* states, const float* actions, const float* advantages);

    // [NEW] Foundational NPP Training (Slow Base Intelligence)
    // Updates the expert pool based on prediction surprise (Entropy)
    void refine_foundation(const float* state, const float* next_bit_target, float lr_scale = 1.0f);

    void reset_state();
    WeightRegistry get_weight_registry();

    // [NEW] Silicon Telemetry: Collapses the wavefront into a 16x16 saliency grid
    void get_saliency_heatmap(float* heatmap_out);

private:
    ::nca::config::EngineConfig engine_cfg_;
    size_t obs_dim_;
    size_t act_dim_;
    ImportanceClassifier classifier_;
    
    // [HARDENED] Silicon-Aligned Buffers (Zero Allocations)
    alignas(64) size_t routing_buffer_[1024]; 
    ::nca::simd::aligned_unique_ptr<float[]> batch_state_pool_; 

    // ── SILICON MODULES ────────────────────────────────────────────────────
    SiliconWeights weights_;
    std::unique_ptr<SiliconWavefront> primary_wavefront_;
    
    std::unique_ptr<::nca::linalg::HashedRouter> router_;
    ::nca::encoding::SiliconEncoder encoder_;
    ::nca::encoding::SiliconVisionEncoder vision_encoder_;

    std::unique_ptr<::nca::spectral::KroneckerRLSState> spectral_rls_;
};

} // namespace nca::execution
