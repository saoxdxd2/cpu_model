// ============================================================================
// NCA — AVX-512 VNNI Kernels
// core/simd/avx512_vnni.cpp
// ============================================================================

#include "core/simd/avx512_vnni.hpp"
#include <immintrin.h>

namespace nca::simd::avx512 {

float vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w, const nca::linalg::MXUINT8Tensor* __restrict x) {
    float global_sum = 0.0f;
    size_t b = 0;
    size_t num_blocks = w->num_blocks;
    
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

} // namespace nca::simd::avx512
