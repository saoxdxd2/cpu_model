// ============================================================================
// NCA — AVX-512 VNNI Kernels
// core/simd/avx512_vnni.cpp
// ============================================================================

#include "core/simd/avx512_vnni.hpp"
#include "core/simd/avx512_math.hpp"
#include <immintrin.h>
#include <algorithm>

namespace nca::simd::avx512 {

float vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w, const nca::linalg::MXUINT8Tensor* __restrict x) {
    const size_t num_blocks = std::min(w->num_blocks, x->num_blocks);
    __m512 v_acc = _mm512_setzero_ps();
    for (size_t b = 0; b < num_blocks; ++b) [[likely]] {
        __m256i vW = _mm256_loadu_si256((const __m256i*)&w->data[b * 32]);
        __m256i vX = _mm256_loadu_si256((const __m256i*)&x->data[b * 32]);
        __m256i i_acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, vW);
        __m128i low = _mm256_castsi256_si128(i_acc);
        __m128i high = _mm256_extracti128_si256(i_acc, 1);
        __m128i combined = _mm_add_epi32(low, high);
        combined = _mm_add_epi32(combined, _mm_srli_si128(combined, 8));
        combined = _mm_add_epi32(combined, _mm_srli_si128(combined, 4));
        float bs = (float)_mm_cvtsi128_si32(combined) - 128.0f * w->w_sums[b];
        float scale = nca::linalg::decode_e8m0_scale(w->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
        v_acc = _mm512_fmadd_ps(_mm512_set1_ps(bs), _mm512_set1_ps(scale), v_acc);
    }
    return _mm512_reduce_add_ps(v_acc);
}

void dual_vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w0, const nca::linalg::MXINT8Tensor* __restrict w1, const nca::linalg::MXUINT8Tensor* __restrict x, float& out0, float& out1) {
    out0 = vnni_dot(w0, x); out1 = vnni_dot(w1, x);
}

void quad_vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w0, const nca::linalg::MXINT8Tensor* __restrict w1, const nca::linalg::MXINT8Tensor* __restrict w2, const nca::linalg::MXINT8Tensor* __restrict w3, const nca::linalg::MXUINT8Tensor* __restrict x, float& out0, float& out1, float& out2, float& out3) {
    out0 = vnni_dot(w0, x); out1 = vnni_dot(w1, x); out2 = vnni_dot(w2, x); out3 = vnni_dot(w3, x);
}

void mx_quantize_x(const float* __restrict in, nca::linalg::MXUINT8Tensor* __restrict out) {
    __m512i v_128 = _mm512_set1_epi32(128);
    __m512i v_0 = _mm512_setzero_si512(), v_255 = _mm512_set1_epi32(255);
    for (size_t b = 0; b < out->num_blocks; ++b) [[likely]] {
        auto v0 = _mm512_loadu_ps(&in[b*32]), v1 = _mm512_loadu_ps(&in[b*32 + 16]);
        float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        out->scales[b] = nca::linalg::extract_e8m0(max_abs);
        auto v_inv = _mm512_set1_ps(1.0f / nca::linalg::decode_e8m0_scale(out->scales[b]));
        auto q = [&](auto v, int off) {
            auto vi = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v, v_inv)), v_128);
            vi = _mm512_min_epi32(_mm512_max_epi32(vi, v_0), v_255);
            _mm_storeu_si128((__m128i*)&out->data[b*32+off], _mm512_cvtepi32_epi8(vi));
        };
        q(v0, 0); q(v1, 16);
    }
}

void mx_fused_silu_quantize_x(const float* __restrict in, nca::linalg::MXUINT8Tensor* __restrict out) {
    __m512i v_128 = _mm512_set1_epi32(128);
    __m512i v_0 = _mm512_setzero_si512(), v_255 = _mm512_set1_epi32(255);
    for (size_t b = 0; b < out->num_blocks; ++b) [[likely]] {
        auto v0 = nca::simd::avx512::silu_ps(_mm512_loadu_ps(&in[b*32])), v1 = nca::simd::avx512::silu_ps(_mm512_loadu_ps(&in[b*32+16]));
        float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        out->scales[b] = nca::linalg::extract_e8m0(max_abs);
        auto v_inv = _mm512_set1_ps(1.0f / nca::linalg::decode_e8m0_scale(out->scales[b]));
        auto q = [&](auto v, int off) {
            auto vi = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v, v_inv)), v_128);
            vi = _mm512_min_epi32(_mm512_max_epi32(vi, v_0), v_255);
            _mm_storeu_si128((__m128i*)&out->data[b*32+off], _mm512_cvtepi32_epi8(vi));
        };
        q(v0, 0); q(v1, 16);
    }
}

} // namespace nca::simd::avx512
