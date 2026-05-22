#pragma once
#include "core/linalg/mx_linear.hpp"
#include <torch/torch.h>
#include <vector>
#include <string>

namespace nca::training {

/**
 * DeepTranslator
 * Low-level primitives for "Silicon-Level" weight translation.
 * Maps foundation data structures to NCA's saturated hardware format.
 */
class DeepTranslator {
public:
    // --- 1. KRONECKER DECOMPOSITION ---
    // Decomposes a large weight W (e.g., 2048x2048) into A (64x64) and B (32x32)
    // factors for the Spectral RLS logic: W ≈ A ⊗ B
    static void decompose_to_kronecker(const torch::Tensor& W, 
                                        float* A_out, float* B_out);

    // --- 2. RANK-16 EXPERT SWIZZLING ---
    // Maps standard MLP layers to NCA's Saliency-Driven Multi-Expert System (SDMS)
    // Handles the Rank-16 alignment and Gated-MLP (SwiGLU) splitting.
    static void swizzle_to_micro_experts(const torch::Tensor& gate_proj,
                                         const torch::Tensor& up_proj,
                                         std::vector<nca::linalg::MXINT8Tensor>& gate_pool,
                                         std::vector<nca::linalg::MXINT8Tensor>& up_pool);

    // --- 3. HARDWARE-LEVEL QUANTIZATION ---
    // Direct control over the MX-Quantization process for custom tuning
    static void manual_mx_quantize(const float* src, 
                                   nca::linalg::MXINT8Tensor& target,
                                   float forced_scale = -1.0f);

    // --- 4. ATTENTION RE-MAPPING ---
    // Translates Multi-Head/Grouped-Query Attention weights into NCA's
    // recursive GLR and Spectral factors.
    static void adopt_attention_block(const torch::Tensor& q_proj,
                                      const torch::Tensor& k_proj,
                                      const torch::Tensor& v_proj,
                                      float* glr_alpha,
                                      float* glr_beta,
                                      size_t d_model);
};

} // namespace nca::training
