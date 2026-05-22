// ============================================================================
// NCA — MX Block-Quantized Linear Algebra
// core/linalg/mx_linear.cpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/avx512_vnni.hpp"
#include <cmath>

namespace nca::linalg {

// ── COMPLEXITY DESTRUCTION: CONDENSED KERNELS ───────────────────────────────

void mx_quantize_w(const float* __restrict in, MXINT8Tensor& out) {
    const size_t num_blocks = out.num_blocks;
    for (size_t b = 0; b < num_blocks; ++b) {
        // Enforce Saturated AVX-512 Quantization
        __m512 v0 = _mm512_loadu_ps(in + b * 32);
        __m512 v1 = _mm512_loadu_ps(in + b * 32 + 16);
        
        float ma = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        out.scales[b] = extract_e8m0(ma);
        
        float inv = 1.0f / (decode_e8m0_scale(out.scales[b]) + 1e-8f);
        __m512 v_inv = _mm512_set1_ps(inv);
        
        auto q = [&](__m512 v, int offset) {
            __m512i vi = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v, v_inv), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
            vi = _mm512_min_epi32(_mm512_max_epi32(vi, _mm512_set1_epi32(-127)), _mm512_set1_epi32(127));
            _mm_storeu_si128((__m128i*)&out.data[b * 32 + offset], _mm512_cvtepi32_epi8(vi));
            return _mm512_reduce_add_epi32(vi);
        };

        out.w_sums[b] = q(v0, 0) + q(v1, 16);
    }
}

void mx_quantize_x(const float* __restrict in, MXUINT8Tensor& out) {
    simd::avx512::mx_quantize_x(in, &out);
}

void mx_fused_silu_quantize_x(const float* __restrict in, MXUINT8Tensor& out) {
    simd::avx512::mx_fused_silu_quantize_x(in, &out);
}

float mx_compute_activation_norm(const MXUINT8Tensor& x_q) {
    return simd::avx512::mx_compute_activation_norm(x_q);
}

void mx_update_gaussian_moment(MXINT8Tensor& w, const MXUINT8Tensor& x_q, float error, float lr, float precomputed_norm) {
    simd::avx512::mx_update_gaussian_moment(w, x_q, error, lr, precomputed_norm);
}

float mx_dot(const MXINT8Tensor& w, const MXUINT8Tensor& x) {
    return simd::avx512::vnni_dot(&w, &x);
}

void mx_dual_dot(const MXINT8Tensor& w0, const MXINT8Tensor& w1, const MXUINT8Tensor& x, float& o0, float& o1) {
    o0 = mx_dot(w0, x); o1 = mx_dot(w1, x);
}

void mx_quad_dot(const MXINT8Tensor& w0, const MXINT8Tensor& w1, const MXINT8Tensor& w2, const MXINT8Tensor& w3, const MXUINT8Tensor& x, float& o0, float& o1, float& o2, float& o3) {
    o0 = mx_dot(w0, x); o1 = mx_dot(w1, x); o2 = mx_dot(w2, x); o3 = mx_dot(w3, x);
}

void mx_rank16_dot(const MXINT8Tensor* __restrict weights, const MXUINT8Tensor& x, float* __restrict outputs) {
    simd::avx512::rank16_vnni_dot(weights, &x, outputs);
}

void mx_rank16_dot_ptrs(const MXINT8Tensor** __restrict weights, const MXUINT8Tensor& x, float* __restrict outputs) {
    simd::avx512::rank16_vnni_dot_ptrs(weights, &x, outputs);
}

void mx_gemv(const MXINT8Tensor& W, const MXUINT8Tensor& x, float* y, size_t rows, size_t cols) {
    const size_t bpr = cols / 32;
    size_t r = 0;
    for (; r + 15 < rows; r += 16) [[likely]] {
        MXINT8Tensor vs[16];
        for(int i=0; i<16; ++i) {
            vs[i].data = (int8_t*)W.data + (r+i)*cols;
            vs[i].scales = W.scales + (r+i)*bpr;
            vs[i].w_sums = W.w_sums + (r+i)*bpr;
            vs[i].num_blocks = bpr;
        }
        mx_rank16_dot(vs, x, y + r);
        for(int i=0; i<16; ++i) vs[i].data = nullptr; 
    }
    for (; r < rows; ++r) {
        MXINT8Tensor v; v.data = (int8_t*)W.data + r*cols; v.scales = W.scales + r*bpr; v.w_sums = W.w_sums + r*bpr; v.num_blocks = bpr;
        y[r] = mx_dot(v, x); v.data = nullptr;
    }
}

} // namespace nca::linalg
