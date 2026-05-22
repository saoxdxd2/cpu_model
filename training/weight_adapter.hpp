#pragma once
#include "core/linalg/mx_linear.hpp"
#include <torch/torch.h>
#include <vector>
#include <string>

namespace nca::execution { class MultimodalEngine; }

namespace nca::training {

enum class NCATarget {
    VisionEncoder,
    StateRecurrence,
    ExpertPoolGate,
    ExpertPoolUp,
    HaltingGate,
    SpectralFactors
};

/**
 * Universal Structural Mapper
 * Handles the recursive translation of external foundation weights 
 * into the NCA's saturated hardware format.
 */
class WeightAdapter {
public:
    // Surgical Injection: Maps a specific external tensor to an NCA component
    // Handles automated shape-matching (padding/cropping) and re-quantization.
    static void inject_tensor(const torch::Tensor& source, 
                              nca::linalg::MXINT8Tensor& target,
                              float gain = 1.0f);

    // Bias/Param Injection: For non-quantized high-precision constants (FP32)
    static void inject_params(const torch::Tensor& source, 
                              float* target_ptr, 
                              size_t target_size);

    // Adoption Registry: Maps an entire set of tensors to the NCA MultimodalEngine
    // This uses structural heuristics to find compatible layers in the base library.
    static void adopt_state_dict(const std::map<std::string, torch::Tensor>& dict, 
                                 nca::execution::MultimodalEngine& engine);

    // Expert Swizzling: Distributes a large MLP or Transformer block across 
    // the NCA SDMS (Saliency-Driven Multi-expert System).
    static void swizzle_to_experts(const torch::Tensor& large_weight, 
                                   std::vector<nca::linalg::MXINT8Tensor>& experts);
};

} // namespace nca::training
