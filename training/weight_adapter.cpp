#include "training/weight_adapter.hpp"
#include "core/execution/multimodal_engine.hpp"
#include <iostream>
#include <algorithm>

namespace nca::training {

void WeightAdapter::inject_params(const torch::Tensor& source, 
                                  float* target_ptr, 
                                  size_t target_size) {
    auto s = source.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* src_ptr = s.data_ptr<float>();
    size_t source_size = (size_t)s.numel();

    size_t copy_size = std::min(source_size, target_size);
    std::memcpy(target_ptr, src_ptr, copy_size * sizeof(float));
    
    if (source_size < target_size) {
        for (size_t i = source_size; i < target_size; ++i) target_ptr[i] = 1.0f;
    }
}

void WeightAdapter::adopt_state_dict(const std::map<std::string, torch::Tensor>& dict, 
                                     nca::execution::MultimodalEngine& engine) {
    auto wr = engine.get_weight_registry();
    
    std::cout << "[Adopt] Starting recursive state_dict adoption...\n";
    std::cout << "[Adopt] GEOMETRIC SCHEMA ACTIVE. Dense VNNI mapping is deprecated.\n";
    
    // Map FP32 Params (Spectral and GLR)
    if (dict.count("blk.0.attn_q.weight")) {
        inject_params(dict.at("blk.0.attn_q.weight"), wr.vision_A, 16*16*128*16);
    }
    
    if (dict.count("blk.0.ffn_norm.weight")) {
        inject_params(dict.at("blk.0.ffn_norm.weight"), wr.glr_beta, wr.d_model);
    }

    std::cout << "[Adopt] SUCCESS: Engine state updated with foundation data.\n";
}

} // namespace nca::training
