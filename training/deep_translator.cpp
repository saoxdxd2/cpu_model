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

    // Legacy MX logic removed during Geometric Schema Migration

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
