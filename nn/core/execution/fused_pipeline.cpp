// ============================================================================
// NCA -- Fused Multimodal Pipeline
// core/execution/fused_pipeline.cpp
// ============================================================================

#include "core/execution/fused_pipeline.hpp"
#include "core/layers/ssm.hpp"
#include "core/simd/dispatch.hpp"
#include <vector>

namespace nca::execution {

void multimodal_fused_step(
    const float* __restrict img_in,
    float* __restrict output,
    nca::vision::ScannerConfig v_cfg,
    const nca::linalg::MXINT8Tensor& W_gate,
    const nca::linalg::MXINT8Tensor& W_up,
    const nca::linalg::MXINT8Tensor& W_down,
    float* __restrict h_ssm,
    const float* __restrict A_ssm,
    const float* __restrict B_ssm,
    const float* __restrict C_ssm
) {
    const auto d_inner = v_cfg.H * v_cfg.W * v_cfg.C;
    std::vector<float> conv_out(d_inner);
    
    nca::linalg::MXUINT8Tensor hidden_q(d_inner / 32);
    nca::layers::SSMConfig ssm_cfg{d_inner, 16};
    
    nca::layers::mx_fused_ssm_silu_quantize_step(
        h_ssm, A_ssm, B_ssm, C_ssm, 
        conv_out.data(), 
        hidden_q, 
        ssm_cfg
    );

    nca::linalg::mx_gemv(W_down, hidden_q, output, W_down.num_blocks * 32 / d_inner, d_inner);
}

} // namespace nca::execution
