#pragma once
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

    // Legacy MX logic removed during Geometric Schema Migration

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
