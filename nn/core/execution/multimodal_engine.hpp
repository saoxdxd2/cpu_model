#pragma once
// ============================================================================
// NCA — Saturated Multimodal Inference Engine (v31.0 - Hardened)
// core/execution/multimodal_engine.hpp
// ============================================================================

#include "core/simd/memory.hpp"
#include "encoding/state_encoder.hpp"
#include "encoding/vision_encoder.hpp"
#include "core/spectral/kronecker_rls.hpp"
#include "config/model_config.hpp"
#include "nn/core/execution/silicon_memory.hpp"
#include "core/execution/wavefront_router.hpp"
#include <memory>
#include <vector>

namespace nca::training { class WeightAdapter; }

namespace nca::execution {

struct WeightRegistry {
    float* vision_A;
    float* vision_B;
    float* vision_C;
    float* glr_alpha;
    float* glr_beta;
    size_t d_model;
};

class MultimodalEngine {
public:
    friend class ::nca::training::WeightAdapter;
    explicit MultimodalEngine(size_t obs_dim = ::nca::config::D_MODEL, 
                              size_t act_dim = 80, 
                              ::nca::config::EngineConfig engine_cfg = {});

    // [NEW] Geometric Schema Inference
    // Bypasses the VNNI matrix math entirely and uses the top-16 structural pointers
    void step_geometric(const float* text_in, const float* image_in, float* out, float temperature = 0.2f);
    void step_geometric_env(const float* obs_in, float* out, float temperature = 0.2f);
    
    // [NEW] Silicon Swarm: Chained Recurrence
    // Output of agent N becomes the context for agent N+1
    void step_swarm(const float* initial_input, float* swarm_out, size_t max_agents);

    void reset_state();
    WeightRegistry get_weight_registry();
    const float* get_state() const { return primary_wavefront_->state.get(); }

    // [NEW] Silicon Telemetry: Collapses the wavefront into a 16x16 saliency grid
    void get_saliency_heatmap(float* heatmap_out);

private:
    ::nca::config::EngineConfig engine_cfg_;
    size_t obs_dim_;
    size_t act_dim_;
    
    // ── SILICON MODULES ────────────────────────────────────────────────────
    SiliconWeights weights_;
    std::unique_ptr<SiliconWavefront> primary_wavefront_;
    
    std::unique_ptr<::nca::execution::WavefrontRouter> geometric_router_;
    
    ::nca::encoding::SiliconEncoder encoder_;
    ::nca::encoding::SiliconVisionEncoder vision_encoder_;

    std::unique_ptr<::nca::spectral::KroneckerRLSState> spectral_rls_;
};

} // namespace nca::execution
