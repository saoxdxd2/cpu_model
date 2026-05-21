#pragma once
// ============================================================================
// NCA — Fused Multimodal Inference Engine
// core/execution/multimodal_engine.hpp
// ============================================================================

#include "core/simd/memory.hpp"
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
    // Internal Aligned Buffers (L1-hot)
    nca::simd::aligned_unique_ptr<float[]> state_;
    nca::simd::aligned_unique_ptr<float[]> vision_latent_;
    
    // Recursive State
    nca::simd::aligned_unique_ptr<float[]> h_glr_;
    nca::simd::aligned_unique_ptr<float[]> h_ssm_;
};

} // namespace nca::execution
