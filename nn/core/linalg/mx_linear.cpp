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
#if defined(__AVX2__) || defined(_MSC_VER)
        __m256 v0 = _mm256_loadu_ps(in + b * 32);
        __m256 v1 = _mm256_loadu_ps(in + b * 32 + 8);
        __m256 v2 = _mm256_loadu_ps(in + b * 32 + 16);
        __m256 v3 = _mm256_loadu_ps(in + b * 32 + 24);
        
        __m256 abs0 = _mm256_max_ps(v0, _mm256_sub_ps(_mm256_setzero_ps(), v0));
        __m256 abs1 = _mm256_max_ps(v1, _mm256_sub_ps(_mm256_setzero_ps(), v1));
        __m256 abs2 = _mm256_max_ps(v2, _mm256_sub_ps(_mm256_setzero_ps(), v2));
        __m256 abs3 = _mm256_max_ps(v3, _mm256_sub_ps(_mm256_setzero_ps(), v3));
        
        __m256 max_val = _mm256_max_ps(_mm256_max_ps(abs0, abs1), _mm256_max_ps(abs2, abs3));
        
        __m128 max_high = _mm256_extractf128_ps(max_val, 1);
        __m128 max_low = _mm256_castps256_ps128(max_val);
        max_low = _mm_max_ps(max_low, max_high);
        max_low = _mm_max_ps(max_low, _mm_movehl_ps(max_low, max_low));
        max_low = _mm_max_ps(max_low, _mm_shuffle_ps(max_low, max_low, _MM_SHUFFLE(1, 1, 1, 1)));
        float ma = _mm_cvtss_f32(max_low);
        
        out.scales[b] = extract_e8m0(ma);
        float inv = 1.0f / decode_e8m0_scale(out.scales[b]);
        __m256 v_inv = _mm256_set1_ps(inv);
        
        __m256i q0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, v_inv));
        __m256i q1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, v_inv));
        __m256i q2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, v_inv));
        __m256i q3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, v_inv));
        
        __m256i min_val = _mm256_set1_epi32(-127);
        __m256i max_val_i = _mm256_set1_epi32(127);
        
        q0 = _mm256_min_epi32(_mm256_max_epi32(q0, min_val), max_val_i);
        q1 = _mm256_min_epi32(_mm256_max_epi32(q1, min_val), max_val_i);
        q2 = _mm256_min_epi32(_mm256_max_epi32(q2, min_val), max_val_i);
        q3 = _mm256_min_epi32(_mm256_max_epi32(q3, min_val), max_val_i);
        
        __m256i p16_01 = _mm256_packs_epi32(q0, q1);
        __m256i p16_23 = _mm256_packs_epi32(q2, q3);
        __m256i p8 = _mm256_packs_epi16(p16_01, p16_23);
        
        __m256i perm_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0); 
        p8 = _mm256_permutevar8x32_epi32(p8, perm_mask);
        
        _mm256_storeu_si256((__m256i*)&out.data[b*32], p8);
        
        __m256i sum_v = _mm256_add_epi32(_mm256_add_epi32(q0, q1), _mm256_add_epi32(q2, q3));
        __m128i sum_high = _mm256_extracti128_si256(sum_v, 1);
        __m128i sum_low = _mm256_castsi256_si128(sum_v);
        sum_low = _mm_add_epi32(sum_low, sum_high);
        sum_low = _mm_hadd_epi32(sum_low, sum_low);
        sum_low = _mm_hadd_epi32(sum_low, sum_low);
        out.w_sums[b] = _mm_cvtsi128_si32(sum_low);
#else
        float ma = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float val = in[b * 32 + i];
            float abs_val = val < 0.0f ? -val : val;
            ma = abs_val > ma ? abs_val : ma;
        }
        out.scales[b] = extract_e8m0(ma);
        float inv = 1.0f / decode_e8m0_scale(out.scales[b]);
        int32_t s = 0;
        for (int i = 0; i < 32; ++i) {
            float scaled = in[b * 32 + i] * inv;
            int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
            int32_t c1 = r < -127 ? -127 : r;
            int32_t c2 = c1 > 127 ? 127 : c1;
            int8_t q = (int8_t)c2;
            out.data[b * 32 + i] = q;
            s += q;
        }
        out.w_sums[b] = s;
#endif
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
