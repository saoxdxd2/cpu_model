// ============================================================================
// NCA — AVX-512 VNNI Kernels
// core/simd/avx512_vnni.cpp
// ============================================================================

#include "core/simd/avx512_vnni.hpp"
#include "core/simd/avx512_math.hpp"
#include <immintrin.h>

namespace nca::simd::avx512 {

float vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w, const nca::linalg::MXUINT8Tensor* __restrict x) {
    float global_sum = 0.0f;
    size_t b = 0;
    size_t num_blocks = std::min(w->num_blocks, x->num_blocks);
    
    for (; b <= num_blocks - 2; b += 2) [[likely]] {
        __m512i vW = _mm512_load_si512((const __m512i*)&w->data[b * 32]);
        __m512i vX = _mm512_load_si512((const __m512i*)&x->data[b * 32]);
        
        __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vX, vW);
        __m512 f_acc = _mm512_cvtepi32_ps(acc);
        
        // Zero-point correction: (x_uint - 128) * w = x_uint*w - 128*w
        // acc contains x_uint*w. We subtract 16 * sum(w) from each of the 8 accumulators to equal 128*sum(w)
        __m256 v_corr0 = _mm256_set1_ps(16.0f * w->w_sums[b]);
        __m256 v_corr1 = _mm256_set1_ps(16.0f * w->w_sums[b+1]);
        __m512 v_corr = _mm512_insertf32x8(_mm512_castps256_ps512(v_corr0), v_corr1, 1);
        f_acc = _mm512_sub_ps(f_acc, v_corr);
        
        float s0 = nca::linalg::decode_e8m0_scale(w->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
        float s1 = nca::linalg::decode_e8m0_scale(w->scales[b+1]) * nca::linalg::decode_e8m0_scale(x->scales[b+1]);
        
        __m256 v_s0 = _mm256_set1_ps(s0);
        __m256 v_s1 = _mm256_set1_ps(s1);
        __m512 v_scales = _mm512_insertf32x8(_mm512_castps256_ps512(v_s0), v_s1, 1);
        
        f_acc = _mm512_mul_ps(f_acc, v_scales);
        global_sum += _mm512_reduce_add_ps(f_acc);
    }
    
    if (b < num_blocks) [[unlikely]] {
        __m256i vW = _mm256_load_si256((const __m256i*)&w->data[b * 32]);
        __m256i vX = _mm256_load_si256((const __m256i*)&x->data[b * 32]);
        __m256i acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, vW);
        __m256 f_acc = _mm256_cvtepi32_ps(acc);
        
        __m256 v_corr = _mm256_set1_ps(16.0f * w->w_sums[b]);
        f_acc = _mm256_sub_ps(f_acc, v_corr);
        
        float s = nca::linalg::decode_e8m0_scale(w->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
        f_acc = _mm256_mul_ps(f_acc, _mm256_set1_ps(s));
        
        __m128 v_low = _mm_add_ps(_mm256_castps256_ps128(f_acc), _mm256_extractf128_ps(f_acc, 1));
        v_low = _mm_hadd_ps(v_low, v_low);
        v_low = _mm_hadd_ps(v_low, v_low);
        global_sum += _mm_cvtss_f32(v_low);
    }
    
    return global_sum;
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
