#include "training/weight_adapter.hpp"
#include "core/execution/multimodal_engine.hpp"
#include <iostream>
#include <algorithm>

namespace nca::training {

void WeightAdapter::inject_tensor(const torch::Tensor& source, 
                                  nca::linalg::MXINT8Tensor& target,
                                  float gain) {
    // 1. Structural Normalization
    auto s = source.to(torch::kCPU).to(torch::kFloat32).contiguous();
    
    // 2. Shape Adaptation (The 'Tuning' part)
    // If source is 2D (Linear), we might need to transpose or flatten
    if (s.dim() > 1) s = s.flatten();
    
    size_t source_size = s.numel();
    size_t target_size = target.num_blocks * 32;

    auto temp = std::vector<float>(target_size, 0.0f);
    const float* src_ptr = s.data_ptr<float>();

    // Copy with gain, padding if source is smaller, cropping if larger
    size_t copy_size = std::min(source_size, target_size);
    for (size_t i = 0; i < copy_size; ++i) {
        temp[i] = src_ptr[i] * gain;
    }

    // 3. NCA MX-Quantization (The 'Hardware Compatibility' part)
    nca::linalg::mx_quantize_w(temp.data(), target);
    
    std::cout << "  [Inject] Mapped foundation data (" << source_size 
              << " units) -> NCA Hardware Blocks (" << target_size << " units)\n";
}

void WeightAdapter::inject_params(const torch::Tensor& source, 
                                  float* target_ptr, 
                                  size_t target_size) {
    auto s = source.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* src_ptr = s.data_ptr<float>();
    size_t source_size = s.numel();

    size_t copy_size = std::min(source_size, target_size);
    std::memcpy(target_ptr, src_ptr, copy_size * sizeof(float));
    
    if (source_size < target_size) {
        // Initialize remainder if source is smaller
        for (size_t i = source_size; i < target_size; ++i) target_ptr[i] = 1.0f;
    }
}

void WeightAdapter::adopt_state_dict(const std::map<std::string, torch::Tensor>& dict, 
                                     nca::execution::MultimodalEngine& engine) {
    auto wr = engine.get_weight_registry();
    
    std::cout << "[Adopt] Starting recursive state_dict adoption...\n";

    // 1. Map Expert Pool (Gated MLP chunks)
    for (size_t i = 0; i < wr.n_experts; ++i) {
        std::string gate_key = "blk." + std::to_string(i/32) + ".ffn_gate.weight"; // Heuristic
        std::string up_key = "blk." + std::to_string(i/32) + ".ffn_up.weight";
        
        if (dict.count(gate_key)) {
            inject_tensor(dict.at(gate_key), *wr.expert_pool_gate[i]);
        }
        if (dict.count(up_key)) {
            inject_tensor(dict.at(up_key), *wr.expert_pool_up[i]);
        }
    }

    // 2. Map FP32 Params (Spectral and GLR)
    if (dict.count("blk.0.attn_q.weight")) {
        inject_params(dict.at("blk.0.attn_q.weight"), wr.vision_A, 16*16*128*16);
    }
    
    if (dict.count("blk.0.ffn_norm.weight")) {
        inject_params(dict.at("blk.0.ffn_norm.weight"), wr.glr_beta, wr.d_model);
    }

    std::cout << "[Adopt] SUCCESS: Engine state updated with foundation data.\n";
}

void WeightAdapter::swizzle_to_experts(const torch::Tensor& large_weight, 
                                       std::vector<nca::linalg::MXINT8Tensor>& experts) {
    auto s = large_weight.to(torch::kCPU).to(torch::kFloat32).contiguous();
    size_t total_rows = s.size(0);
    size_t d_model = s.size(1);
    
    const float* src_ptr = s.data_ptr<float>();
    
    // NCA Expert Pool usually has 1024 micro-experts of size 16xD
    size_t experts_to_fill = std::min(experts.size(), total_rows / 16);
    
    std::cout << "  [Swizzle] Distributing foundation layer across " << experts_to_fill << " NCA Experts...\n";

    for (size_t i = 0; i < experts_to_fill; ++i) {
        // Slice a 16-row block for this expert
        // (Simplified: NCA experts are stored as flattened blocks of D/32)
        // We'll use inject_tensor to handle the quantization of each expert's weights
        auto expert_slice = s.slice(0, i * 16, (i + 1) * 16).flatten();
        inject_tensor(expert_slice, experts[i]);
    }
}

} // namespace nca::training
