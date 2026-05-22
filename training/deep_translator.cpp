#include "training/deep_translator.hpp"
#include <iostream>
#include <cmath>

namespace nca::training {

void DeepTranslator::decompose_to_kronecker(const torch::Tensor& W, 
                                            float* A_out, float* B_out) {
    // Structural SVD-based Kronecker Approximation
    // NCA target: A (64x64), B (32x32)
    auto w = W.to(torch::kCPU).to(torch::kFloat32).reshape({32, 64, 32, 64});
    
    // Rearrange to align with Kronecker product structure
    auto reshaped = w.permute({0, 2, 1, 3}).reshape({1024, 4096});
    
    // Compute SVD of the rearranged matrix
    auto svd = torch::linalg_svd(reshaped, false);
    auto U = std::get<0>(svd);
    auto S = std::get<1>(svd);
    auto V = std::get<2>(svd);

    // The first singular value gives the best Kronecker approximation
    float sigma = S[0].item<float>();
    auto vec_b = U.index({torch::indexing::Slice(), 0});
    auto vec_a = V.index({0, torch::indexing::Slice()});

    // Populate B (32x32) and A (64x64)
    std::memcpy(B_out, vec_b.data_ptr<float>(), 1024 * sizeof(float));
    std::memcpy(A_out, vec_a.data_ptr<float>(), 4096 * sizeof(float));
    
    // Apply sqrt(sigma) to both to maintain scale
    float scale = std::sqrt(sigma);
    for(int i=0; i<1024; ++i) B_out[i] *= scale;
    for(int i=0; i<4096; ++i) A_out[i] *= scale;

    std::cout << "  [Kronecker] Decomposed " << W.size(0) << "x" << W.size(1) 
              << " into 64x64 and 32x32 factors (Appx Error: " 
              << (S.sum() - S[0]).item<float>() << ")\n";
}

void DeepTranslator::swizzle_to_micro_experts(const torch::Tensor& gate_proj,
                                              const torch::Tensor& up_proj,
                                              std::vector<nca::linalg::MXINT8Tensor>& gate_pool,
                                              std::vector<nca::linalg::MXINT8Tensor>& up_pool) {
    auto g = gate_proj.to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto u = up_proj.to(torch::kCPU).to(torch::kFloat32).contiguous();
    
    size_t rows = g.size(0);
    size_t cols = g.size(1);
    
    // Gemma/Llama use large SwiGLU MLPs (e.g., 16384 -> 2048)
    // We swizzle these into 1024 Micro-Experts (16 rows each)
    size_t count = std::min(gate_pool.size(), rows / 16);
    
    for (size_t i = 0; i < count; ++i) {
        auto g_slice = g.slice(0, i * 16, (i + 1) * 16).flatten();
        auto u_slice = u.slice(0, i * 16, (i + 1) * 16).flatten();
        
        nca::linalg::mx_quantize_w(g_slice.data_ptr<float>(), gate_pool[i]);
        nca::linalg::mx_quantize_w(u_slice.data_ptr<float>(), up_pool[i]);
    }
    
    std::cout << "  [Swizzle] Ported SwiGLU layers to " << count << " Rank-16 NCA experts.\n";
}

void DeepTranslator::manual_mx_quantize(const float* src, 
                                        nca::linalg::MXINT8Tensor& target,
                                        float forced_scale) {
    // Low-level bridge to the engine's quantization kernel
    nca::linalg::mx_quantize_w(src, target);
    
    if (forced_scale > 0) {
        // Override computed scales for manual hardware tuning
        for (size_t i = 0; i < target.num_blocks; ++i) {
            target.scales[i] = static_cast<uint8_t>(std::clamp(std::log2(forced_scale) + 127.0f, 0.0f, 255.0f));
        }
    }
}

void DeepTranslator::adopt_attention_block(const torch::Tensor& q_proj,
                                           const torch::Tensor& k_proj,
                                           const torch::Tensor& v_proj,
                                           float* glr_alpha,
                                           float* glr_beta,
                                           size_t d_model) {
    // Translation logic to map Attention dynamics to Grounded Recurrence (GLR)
    // Beta (Input gain) is derived from Query/Key signal strength
    auto q_f32 = q_proj.to(torch::kFloat32);
    auto k_f32 = k_proj.to(torch::kFloat32);
    
    // Compute Frobenius norm (default for linalg_norm without ord)
    auto q_norm = torch::linalg_norm(q_f32).item<float>();
    auto k_norm = torch::linalg_norm(k_f32).item<float>();
    
    float beta_avg = std::sqrt(q_norm * k_norm) * 0.01f;
    for (size_t i = 0; i < d_model; ++i) {
        glr_alpha[i] = 0.999f; // Stability anchor
        glr_beta[i]  = beta_avg;
    }
    
    std::cout << "  [Attention] Adopted QKV dynamics into GLR factors (Beta: " << beta_avg << ")\n";
}

} // namespace nca::training
