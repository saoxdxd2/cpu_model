#pragma once
// ============================================================================
// NCA — AVX-512 VNNI Kernels
// core/simd/avx512_vnni.hpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"

namespace nca::simd::avx512 {

// Computes dot product between E8M0 int8 weights and uint8 activations using VNNI instructions
float vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w, const nca::linalg::MXUINT8Tensor* __restrict x);

// Dual VNNI: Computes two dot products simultaneously to share 'x' in L1.
void dual_vnni_dot(
    const nca::linalg::MXINT8Tensor* __restrict w0,
    const nca::linalg::MXINT8Tensor* __restrict w1,
    const nca::linalg::MXUINT8Tensor* __restrict x,
    float& out0,
    float& out1
);

// Quad VNNI: Computes four dot products simultaneously, sharing 'x' 4x.
void quad_vnni_dot(
    const nca::linalg::MXINT8Tensor* __restrict w0,
    const nca::linalg::MXINT8Tensor* __restrict w1,
    const nca::linalg::MXINT8Tensor* __restrict w2,
    const nca::linalg::MXINT8Tensor* __restrict w3,
    const nca::linalg::MXUINT8Tensor* __restrict x,
    float& out0,
    float& out1,
    float& out2,
    float& out3
);

void rank16_vnni_dot(const nca::linalg::MXINT8Tensor* __restrict weights, const nca::linalg::MXUINT8Tensor* __restrict x, float* __restrict outputs);
void rank16_vnni_dot_ptrs(const nca::linalg::MXINT8Tensor** __restrict weights, const nca::linalg::MXUINT8Tensor* __restrict x, float* __restrict outputs);

void mx_quantize_x(const float* __restrict in, nca::linalg::MXUINT8Tensor* __restrict out);
void mx_fused_silu_quantize_x(const float* __restrict in, nca::linalg::MXUINT8Tensor* __restrict out);

} // namespace nca::simd::avx512
