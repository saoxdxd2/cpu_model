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
    size_t num_blocks = std::min(w->num_blocks, x->num_blocks);
    __m512 v_acc_global = _mm512_setzero_ps();
    
    size_t b = 0;
    for (; b + 2 <= num_blocks; b += 2) [[likely]] {
        __m512i vW = _mm512_stream_load_si512((__m512i*)&w->data[b * 32]);
        __m512i vX = _mm512_load_si512((const __m512i*)&x->data[b * 32]);
        
        __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vX, vW);
        __m512 f_acc = _mm512_cvtepi32_ps(acc);
        
        __m512 v_corr = _mm512_mask_blend_ps(0x00FF, _mm512_setzero_ps(), _mm512_set1_ps(128.0f * w->w_sums[b]));
        v_corr = _mm512_mask_blend_ps(0xFF00, v_corr, _mm512_set1_ps(128.0f * w->w_sums[b+1]));
        f_acc = _mm512_sub_ps(f_acc, v_corr);
        
        float s0 = nca::linalg::decode_e8m0_scale(w->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
        float s1 = nca::linalg::decode_e8m0_scale(w->scales[b+1]) * nca::linalg::decode_e8m0_scale(x->scales[b+1]);
        __m512 v_scales = _mm512_mask_blend_ps(0x00FF, _mm512_setzero_ps(), _mm512_set1_ps(s0));
        v_scales = _mm512_mask_blend_ps(0xFF00, v_scales, _mm512_set1_ps(s1));
        
        v_acc_global = _mm512_fmadd_ps(f_acc, v_scales, v_acc_global);
    }
    
    float global_sum = _mm512_reduce_add_ps(v_acc_global);
    for (; b < num_blocks; ++b) {
        __m256i vW = _mm256_load_si256((const __m256i*)&w->data[b * 32]);
        __m256i vX = _mm256_load_si256((const __m256i*)&x->data[b * 32]);
        __m256i acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, vW);
        __m128i low = _mm256_castsi256_si128(acc);
        __m128i high = _mm256_extracti128_si256(acc, 1);
        __m128i combined = _mm_add_epi32(low, high);
        combined = _mm_add_epi32(combined, _mm_srli_si128(combined, 8));
        combined = _mm_add_epi32(combined, _mm_srli_si128(combined, 4));
        float block_sum = static_cast<float>(_mm_cvtsi128_si32(combined));
        block_sum -= 128.0f * w->w_sums[b];
        block_sum *= nca::linalg::decode_e8m0_scale(w->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
        global_sum += block_sum;
    }
    return global_sum;
}

void dual_vnni_dot(
    const nca::linalg::MXINT8Tensor* __restrict w0,
    const nca::linalg::MXINT8Tensor* __restrict w1,
    const nca::linalg::MXUINT8Tensor* __restrict x,
    float& out0,
    float& out1
) {
    size_t num_blocks = x->num_blocks;
    __m512 v_acc0 = _mm512_setzero_ps();
    __m512 v_acc1 = _mm512_setzero_ps();
    
    size_t b = 0;
    for (; b + 2 <= num_blocks; b += 2) [[likely]] {
        __m512i vX = _mm512_load_si512((const __m512i*)&x->data[b * 32]);
        __m512i vW0 = _mm512_stream_load_si512((__m512i*)&w0->data[b * 32]);
        __m512i vW1 = _mm512_stream_load_si512((__m512i*)&w1->data[b * 32]);
        
        __m512i i_acc0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vX, vW0);
        __m512i i_acc1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vX, vW1);
        
        __m512 f_acc0 = _mm512_cvtepi32_ps(i_acc0);
        __m512 f_acc1 = _mm512_cvtepi32_ps(i_acc1);

        // Vectorized Meta 0
        __m512 v_corr0 = _mm512_mask_blend_ps(0x00FF, _mm512_setzero_ps(), _mm512_set1_ps(128.0f * w0->w_sums[b]));
        v_corr0 = _mm512_mask_blend_ps(0xFF00, v_corr0, _mm512_set1_ps(128.0f * w0->w_sums[b+1]));
        f_acc0 = _mm512_sub_ps(f_acc0, v_corr0);
        float s0_0 = nca::linalg::decode_e8m0_scale(w0->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
        float s0_1 = nca::linalg::decode_e8m0_scale(w0->scales[b+1]) * nca::linalg::decode_e8m0_scale(x->scales[b+1]);
        __m512 v_sc0 = _mm512_mask_blend_ps(0x00FF, _mm512_setzero_ps(), _mm512_set1_ps(s0_0));
        v_sc0 = _mm512_mask_blend_ps(0xFF00, v_sc0, _mm512_set1_ps(s0_1));
        v_acc0 = _mm512_fmadd_ps(f_acc0, v_sc0, v_acc0);

        // Vectorized Meta 1
        __m512 v_corr1 = _mm512_mask_blend_ps(0x00FF, _mm512_setzero_ps(), _mm512_set1_ps(128.0f * w1->w_sums[b]));
        v_corr1 = _mm512_mask_blend_ps(0xFF00, v_corr1, _mm512_set1_ps(128.0f * w1->w_sums[b+1]));
        f_acc1 = _mm512_sub_ps(f_acc1, v_corr1);
        float s1_0 = nca::linalg::decode_e8m0_scale(w1->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
        float s1_1 = nca::linalg::decode_e8m0_scale(w1->scales[b+1]) * nca::linalg::decode_e8m0_scale(x->scales[b+1]);
        __m512 v_sc1 = _mm512_mask_blend_ps(0x00FF, _mm512_setzero_ps(), _mm512_set1_ps(s1_0));
        v_sc1 = _mm512_mask_blend_ps(0xFF00, v_sc1, _mm512_set1_ps(s1_1));
        v_acc1 = _mm512_fmadd_ps(f_acc1, v_sc1, v_acc1);
    }
    
    out0 = _mm512_reduce_add_ps(v_acc0);
    out1 = _mm512_reduce_add_ps(v_acc1);
    
    // Tail
    for (; b < num_blocks; ++b) {
        auto step = [&](const nca::linalg::MXINT8Tensor* w) {
            __m256i vW = _mm256_load_si256((const __m256i*)&w->data[b * 32]);
            __m256i vX = _mm256_load_si256((const __m256i*)&x->data[b * 32]);
            __m256i acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, vW);
            __m128i low = _mm256_castsi256_si128(acc);
            __m128i high = _mm256_extracti128_si256(acc, 1);
            __m128i combined = _mm_add_epi32(low, high);
            combined = _mm_add_epi32(combined, _mm_srli_si128(combined, 8));
            combined = _mm_add_epi32(combined, _mm_srli_si128(combined, 4));
            float sum = static_cast<float>(_mm_cvtsi128_si32(combined));
            sum -= 128.0f * w->w_sums[b];
            sum *= nca::linalg::decode_e8m0_scale(w->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
            return sum;
        };
        out0 += step(w0);
        out1 += step(w1);
    }
}

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
) {
    size_t num_blocks = x->num_blocks;
    __m512 v_acc0 = _mm512_setzero_ps();
    __m512 v_acc1 = _mm512_setzero_ps();
    __m512 v_acc2 = _mm512_setzero_ps();
    __m512 v_acc3 = _mm512_setzero_ps();
    
    size_t b = 0;
    for (; b + 2 <= num_blocks; b += 2) [[likely]] {
        __m512i vX = _mm512_load_si512((const __m512i*)&x->data[b * 32]);
        
        __m512i vW0 = _mm512_stream_load_si512((__m512i*)&w0->data[b * 32]);
        __m512i vW1 = _mm512_stream_load_si512((__m512i*)&w1->data[b * 32]);
        __m512i vW2 = _mm512_stream_load_si512((__m512i*)&w2->data[b * 32]);
        __m512i vW3 = _mm512_stream_load_si512((__m512i*)&w3->data[b * 32]);
        
        auto step = [&](__m512i vW, const nca::linalg::MXINT8Tensor* w, __m512& v_acc) {
            __m512i i_acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vX, vW);
            __m512 f_acc = _mm512_cvtepi32_ps(i_acc);
            __m512 v_corr = _mm512_mask_blend_ps(0x00FF, _mm512_setzero_ps(), _mm512_set1_ps(128.0f * w->w_sums[b]));
            v_corr = _mm512_mask_blend_ps(0xFF00, v_corr, _mm512_set1_ps(128.0f * w->w_sums[b+1]));
            f_acc = _mm512_sub_ps(f_acc, v_corr);
            float s0 = nca::linalg::decode_e8m0_scale(w->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
            float s1 = nca::linalg::decode_e8m0_scale(w->scales[b+1]) * nca::linalg::decode_e8m0_scale(x->scales[b+1]);
            __m512 v_scales = _mm512_mask_blend_ps(0x00FF, _mm512_setzero_ps(), _mm512_set1_ps(s0));
            v_scales = _mm512_mask_blend_ps(0xFF00, v_scales, _mm512_set1_ps(s1));
            v_acc = _mm512_fmadd_ps(f_acc, v_scales, v_acc);
        };

        step(vW0, w0, v_acc0);
        step(vW1, w1, v_acc1);
        step(vW2, w2, v_acc2);
        step(vW3, w3, v_acc3);
    }
    
    out0 = _mm512_reduce_add_ps(v_acc0);
    out1 = _mm512_reduce_add_ps(v_acc1);
    out2 = _mm512_reduce_add_ps(v_acc2);
    out3 = _mm512_reduce_add_ps(v_acc3);
    
    // Tail
    for (; b < num_blocks; ++b) {
        auto tail_step = [&](const nca::linalg::MXINT8Tensor* w) {
            __m256i vW = _mm256_load_si256((const __m256i*)&w->data[b * 32]);
            __m256i vX = _mm256_load_si256((const __m256i*)&x->data[b * 32]);
            __m256i acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, vW);
            __m128i low = _mm256_castsi256_si128(acc);
            __m128i high = _mm256_extracti128_si256(acc, 1);
            __m128i combined = _mm_add_epi32(low, high);
            combined = _mm_add_epi32(combined, _mm_srli_si128(combined, 8));
            combined = _mm_add_epi32(combined, _mm_srli_si128(combined, 4));
            float sum = static_cast<float>(_mm_cvtsi128_si32(combined));
            sum -= 128.0f * w->w_sums[b];
            sum *= nca::linalg::decode_e8m0_scale(w->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
            return sum;
        };
        out0 += tail_step(w0);
        out1 += tail_step(w1);
        out2 += tail_step(w2);
        out3 += tail_step(w3);
    }
}

void mx_quantize_x(const float* __restrict in, nca::linalg::MXUINT8Tensor* __restrict out) {
    __m512i v_sign_mask = _mm512_set1_epi32(0x7FFFFFFF);
    __m512i v_128 = _mm512_set1_epi32(128);
    __m512i v_0 = _mm512_setzero_si512();
    __m512i v_255 = _mm512_set1_epi32(255);

    for (size_t b = 0; b < out->num_blocks; ++b) [[likely]] {
        __m512 v_in0 = _mm512_loadu_ps(&in[b*32]);
        __m512 v_in1 = _mm512_loadu_ps(&in[b*32 + 16]);

        __m512 v_abs0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v_in0), v_sign_mask));
        __m512 v_abs1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v_in1), v_sign_mask));

        float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(v_abs0, v_abs1));

        out->scales[b] = nca::linalg::extract_e8m0(max_abs);
        float scale = nca::linalg::decode_e8m0_scale(out->scales[b]);
        float inv = scale > 0 ? 1.0f / scale : 0.0f;

        __m512 v_inv = _mm512_set1_ps(inv);
        __m512 v_scaled0 = _mm512_mul_ps(v_in0, v_inv);
        __m512 v_scaled1 = _mm512_mul_ps(v_in1, v_inv);

        __m512i v_q0 = _mm512_cvtps_epi32(v_scaled0);
        __m512i v_q1 = _mm512_cvtps_epi32(v_scaled1);

        v_q0 = _mm512_add_epi32(v_q0, v_128);
        v_q1 = _mm512_add_epi32(v_q1, v_128);

        v_q0 = _mm512_min_epi32(_mm512_max_epi32(v_q0, v_0), v_255);
        v_q1 = _mm512_min_epi32(_mm512_max_epi32(v_q1, v_0), v_255);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out->data[b*32]), _mm512_cvtepi32_epi8(v_q0));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out->data[b*32 + 16]), _mm512_cvtepi32_epi8(v_q1));
    }
}

void mx_fused_silu_quantize_x(const float* __restrict in, nca::linalg::MXUINT8Tensor* __restrict out) {
    __m512i v_sign_mask = _mm512_set1_epi32(0x7FFFFFFF);
    __m512i v_128 = _mm512_set1_epi32(128);
    __m512i v_0 = _mm512_setzero_si512();
    __m512i v_255 = _mm512_set1_epi32(255);

    for (size_t b = 0; b < out->num_blocks; ++b) [[likely]] {
        __m512 v_in0 = _mm512_loadu_ps(&in[b*32]);
        __m512 v_in1 = _mm512_loadu_ps(&in[b*32 + 16]);

        // --- KERNEL FUSION: SILU ---
        v_in0 = nca::simd::avx512::silu_ps(v_in0);
        v_in1 = nca::simd::avx512::silu_ps(v_in1);
        // ---------------------------

        __m512 v_abs0 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v_in0), v_sign_mask));
        __m512 v_abs1 = _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(v_in1), v_sign_mask));

        float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(v_abs0, v_abs1));

        out->scales[b] = nca::linalg::extract_e8m0(max_abs);
        float scale = nca::linalg::decode_e8m0_scale(out->scales[b]);
        float inv = scale > 0 ? 1.0f / scale : 0.0f;

        __m512 v_inv = _mm512_set1_ps(inv);
        __m512 v_scaled0 = _mm512_mul_ps(v_in0, v_inv);
        __m512 v_scaled1 = _mm512_mul_ps(v_in1, v_inv);

        __m512i v_q0 = _mm512_cvtps_epi32(v_scaled0);
        __m512i v_q1 = _mm512_cvtps_epi32(v_scaled1);

        v_q0 = _mm512_add_epi32(v_q0, v_128);
        v_q1 = _mm512_add_epi32(v_q1, v_128);

        v_q0 = _mm512_min_epi32(_mm512_max_epi32(v_q0, v_0), v_255);
        v_q1 = _mm512_min_epi32(_mm512_max_epi32(v_q1, v_0), v_255);

        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out->data[b*32]), _mm512_cvtepi32_epi8(v_q0));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out->data[b*32 + 16]), _mm512_cvtepi32_epi8(v_q1));
    }
}

} // namespace nca::simd::avx512
