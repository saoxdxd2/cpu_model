// ============================================================================
// NCA — Fused Inference Pipeline
// core/execution/fused_pipeline.cpp
// ============================================================================

#include "core/execution/fused_pipeline.hpp"
#include "core/simd/avx512_kernels.hpp"
#include "core/linalg/mx_linear.hpp"
#include "config/model_config.hpp"
#include <immintrin.h>

namespace nca::execution {

void multimodal_fused_step(
    const float* __restrict text_in,
    const float* __restrict img_in,
    const nca::linalg::MXINT8Tensor* __restrict W_gate,
    const nca::linalg::MXINT8Tensor* __restrict W_up,
    float* __restrict out
) {
    alignas(64) float conv_out[2048]; // Using D_MODEL=2048
    for(size_t i=0; i<nca::config::D_MODEL; ++i) conv_out[i] = img_in[i] * 0.5f;

    nca::linalg::MXUINT8Tensor x_q(nca::config::D_MODEL);
    nca::linalg::mx_quantize_x(conv_out, x_q);

    alignas(64) float gate_res[2048];
    alignas(64) float up_res[2048];

    for(size_t i=0; i < nca::config::D_MODEL; i += 16) {
        nca::linalg::mx_rank16_dot(&W_gate[i], x_q, &gate_res[i]);
        nca::linalg::mx_rank16_dot(&W_up[i], x_q, &up_res[i]);
    }

    nca::simd::avx512::silu(gate_res, nca::config::D_MODEL);

    for(size_t i=0; i < nca::config::D_MODEL; i += 16) {
        __m512 v_gate = _mm512_load_ps(&gate_res[i]);
        __m512 v_up = _mm512_load_ps(&up_res[i]);
        __m512 v_res = _mm512_mul_ps(v_gate, v_up);
        _mm512_store_ps(&out[i], v_res);
    }
}

} // namespace nca::execution
