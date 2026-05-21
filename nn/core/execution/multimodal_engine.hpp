#pragma once
// ============================================================================
// NCA — Fused Multimodal Inference Engine
// core/execution/multimodal_engine.hpp
// ============================================================================

#include "core/vision/scanner.hpp"
#include "core/vision/spectral_pruner.hpp"
#include "core/execution/latent_adapter.hpp"
#include "core/layers/glr.hpp"
#include "core/layers/ssm.hpp"
#include "core/layers/sla.hpp"
#include "core/layers/halting.hpp"
#include <vector>

namespace nca::execution {

class MultimodalEngine {
public:
    struct Config {
        nca::vision::ScannerConfig vision_cfg;
        nca::vision::SpectralPruner::Config prune_cfg;
        nca::execution::LatentAdapter::Config adapter_cfg;
        size_t d_model;
    };

    explicit MultimodalEngine(Config cfg);

    // Executes the COMPLETE custom neural network logic:
    // Vision (Scan+Prune) -> Bridge (Adapter) -> Logic (Hybrid Reasoning)
    void step(
        const float* __restrict image_in,
        float* __restrict hidden_state, // [d_model]
        float* __restrict logic_output   // [d_model]
    );

private:
    Config cfg_;
    std::unique_ptr<nca::vision::SpectralPruner> pruner_;
    std::unique_ptr<nca::execution::LatentAdapter> adapter_;
    
    // Internal L1-hot buffers
    std::vector<float> scanner_buf_;
    std::vector<size_t> active_indices_;
    std::vector<float> shuffled_buf_;
    
    // Logic Core States
    nca::layers::HaltingState halt_state_;
};

} // namespace nca::execution
