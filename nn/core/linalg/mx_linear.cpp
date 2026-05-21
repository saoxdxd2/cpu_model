// ============================================================================
// NCA — MX Block-Quantized Linear Algebra
// core/linalg/mx_linear.cpp
// ============================================================================

#include "core/linalg/mx_linear.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/avx512_vnni.hpp"
#include <cmath>

namespace nca::linalg {

static float mx_dot_scalar(const MXINT8Tensor& w, const MXUINT8Tensor& x) {
    float sum = 0.0f;
    for (size_t b = 0; b < std::min(w.num_blocks, x.num_blocks); ++b) {
        float b_sum = 0;
        for (int i = 0; i < 32; ++i) b_sum += (float)((int)x.data[b*32+i]-128) * w.data[b*32+i];
        sum += b_sum * decode_e8m0_scale(w.scales[b]) * decode_e8m0_scale(x.scales[b]);
    }
    return sum;
}

void mx_quantize_w(const float* __restrict in, MXINT8Tensor& out) {
    for (size_t b = 0; b < out.num_blocks; ++b) {
        float ma = 0;
        for (int i = 0; i < 32; ++i) ma = std::max(ma, std::abs(in[b*32+i]));
        out.scales[b] = extract_e8m0(ma);
        auto inv = 1.0f / decode_e8m0_scale(out.scales[b]);
        int32_t s = 0;
        for (int i = 0; i < 32; ++i) {
            auto q = (int8_t)std::clamp(std::round(in[b*32+i] * inv), -127.f, 127.f);
            out.data[b*32+i] = q; s += q;
        }
        out.w_sums[b] = s;
    }
}

void mx_quantize_x(const float* __restrict in, MXUINT8Tensor& out) {
    if (simd::best_backend() == simd::Backend::AVX512) simd::avx512::mx_quantize_x(in, &out);
}

void mx_fused_silu_quantize_x(const float* __restrict in, MXUINT8Tensor& out) {
    if (simd::best_backend() == simd::Backend::AVX512) simd::avx512::mx_fused_silu_quantize_x(in, &out);
}

float mx_dot(const MXINT8Tensor& w, const MXUINT8Tensor& x) {
    if (simd::best_backend() == simd::Backend::AVX512) return simd::avx512::vnni_dot(&w, &x);
    return mx_dot_scalar(w, x);
}

void mx_dual_dot(const MXINT8Tensor& w0, const MXINT8Tensor& w1, const MXUINT8Tensor& x, float& o0, float& o1) {
    if (simd::best_backend() == simd::Backend::AVX512) { simd::avx512::dual_vnni_dot(&w0, &w1, &x, o0, o1); return; }
    o0 = mx_dot_scalar(w0, x); o1 = mx_dot_scalar(w1, x);
}

void mx_quad_dot(const MXINT8Tensor& w0, const MXINT8Tensor& w1, const MXINT8Tensor& w2, const MXINT8Tensor& w3, const MXUINT8Tensor& x, float& o0, float& o1, float& o2, float& o3) {
    if (simd::best_backend() == simd::Backend::AVX512) { simd::avx512::quad_vnni_dot(&w0, &w1, &w2, &w3, &x, o0, o1, o2, o3); return; }
    o0 = mx_dot_scalar(w0, x); o1 = mx_dot_scalar(w1, x); o2 = mx_dot_scalar(w2, x); o3 = mx_dot_scalar(w3, x);
}

void mx_rank16_dot(const MXINT8Tensor* __restrict weights, const MXUINT8Tensor& x, float* __restrict outputs) {
    if (simd::best_backend() == simd::Backend::AVX512) { simd::avx512::rank16_vnni_dot(weights, &x, outputs); return; }
    for(int i=0; i<16; ++i) outputs[i] = mx_dot_scalar(weights[i], x);
}

void mx_gemv(const MXINT8Tensor& W, const MXUINT8Tensor& x, float* y, size_t rows, size_t cols) {
    const size_t bpr = cols / 32;
    if (simd::best_backend() == simd::Backend::AVX512) {
        size_t r = 0;
        // ── RANK-16 EXTREME UNROLLING ───────────────────────────────────────
        for (; r + 15 < rows; r += 16) [[likely]] {
            MXINT8Tensor views[16];
            for(int i=0; i<16; ++i) {
                views[i].data = (int8_t*)W.data + (r+i)*cols;
                views[i].scales = W.scales + (r+i)*bpr;
                views[i].w_sums = W.w_sums + (r+i)*bpr;
                views[i].num_blocks = bpr;
            }
            mx_rank16_dot(views, x, y + r);
            for(int i=0; i<16; ++i) views[i].data = nullptr;
        }
        // Quad tail
        for (; r + 3 < rows; r += 4) {
            MXINT8Tensor v0, v1, v2, v3;
            auto p = [&](auto& v, size_t i) { v.data = (int8_t*)W.data + i*cols; v.scales = W.scales + i*bpr; v.w_sums = W.w_sums + i*bpr; v.num_blocks = bpr; };
            p(v0, r); p(v1, r+1); p(v2, r+2); p(v3, r+3);
            mx_quad_dot(v0, v1, v2, v3, x, y[r], y[r+1], y[r+2], y[r+3]);
            v0.data = v1.data = v2.data = v3.data = nullptr;
        }
        // Single tail
        for (; r < rows; ++r) {
            MXINT8Tensor v; v.data = (int8_t*)W.data + r*cols; v.scales = W.scales + r*bpr; v.w_sums = W.w_sums + r*bpr; v.num_blocks = bpr;
            y[r] = mx_dot(v, x); v.data = nullptr;
        }
    } else {
        for (size_t r = 0; r < rows; ++r) {
            float s = 0;
            for (size_t b = 0; b < bpr; ++b) {
                float bs = 0; for (int i = 0; i < 32; ++i) bs += (float)((int)x.data[b*32+i]-128) * W.data[(r*bpr+b)*32+i];
                s += bs * decode_e8m0_scale(W.scales[r*bpr+b]) * decode_e8m0_scale(x.scales[b]);
            }
            y[r] = s;
        }
    }
}

} // namespace nca::linalg
