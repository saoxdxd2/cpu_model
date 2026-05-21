#pragma once
// ============================================================================
// NCA — Fused Multimodal Inference Engine
// core/execution/multimodal_engine.hpp
// ============================================================================

#include "core/simd/memory.hpp"
#include "core/execution/latent_adapter.hpp"
#include "core/linalg/mx_linear.hpp"
#include <memory>
#include <vector>

namespace nca::execution {

class MultimodalEngine {
public:
    MultimodalEngine();

    // Executes the COMPLETE custom neural network logic:
    // Vision (Scan+Prune) -> Bridge (Adapter) -> Recursive Logic (ACT Depth)
    void step(const float* text_in, const float* image_in, float* out);

private:
    // ── PERSISTENT WEIGHT ANCHORS ───────────────────────────────────────────
    // Vision Weights
    nca::simd::aligned_unique_ptr<float[]> W_vision_A_;
    nca::simd::aligned_unique_ptr<float[]> W_vision_B_;
    nca::simd::aligned_unique_ptr<float[]> W_vision_C_;

    // Logic Weights (Backbone)
    nca::simd::aligned_unique_ptr<float[]> W_glr_alpha_;
    nca::simd::aligned_unique_ptr<float[]> W_glr_beta_;
    std::vector<nca::linalg::MXINT8Tensor> W_mlp_gate_;
    std::vector<nca::linalg::MXINT8Tensor> W_mlp_up_;
    
    // Bridge Core
    nca::execution::LatentAdapter adapter_;
    
    // Halting Core
    nca::linalg::MXINT8Tensor W_halt_;

    // ── PERSISTENT STATE BUFFERS (L1-HOT) ───────────────────────────────────
    nca::simd::aligned_unique_ptr<float[]> state_;
    nca::simd::aligned_unique_ptr<float[]> vision_latent_;
    nca::simd::aligned_unique_ptr<float[]> h_glr_;
    nca::simd::aligned_unique_ptr<float[]> h_ssm_;
};

} // namespace nca::execution
