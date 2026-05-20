// ============================================================================
// NCA — AVX-512 VNNI Kernels
// core/simd/avx512_vnni.cpp
//
// Contains the VPDPBUSD core logic for MXINT8 accumulations.
// ============================================================================

#include "core/simd/avx512_vnni.hpp"
#include <immintrin.h>
#include <bit>

namespace nca::simd::avx512 {

inline float decode_scale(uint8_t e) {
    if (e == 0) return 0.0f;
    uint32_t bits = static_cast<uint32_t>(e) << 23;
    return std::bit_cast<float>(bits);
}

float vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w, const nca::linalg::MXUINT8Tensor* __restrict x) {
    float global_sum = 0.0f;
    size_t b = 0;
    size_t num_blocks = w->num_blocks;
    
    // Process 64 elements (2 blocks) per cycle
    // Using _mm512_load_si512 instead of loadu because data is guaranteed 64-byte aligned!
    for (; b <= num_blocks - 2; b += 2) [[likely]] {
        __m512i vW = _mm512_load_si512((const __m512i*)&w->data[b * 32]);
        __m512i vX = _mm512_load_si512((const __m512i*)&x->data[b * 32]);
        
        // VNNI Magic: 64 MACs into 16 32-bit accumulators
        __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), vX, vW);
        __m512 f_acc = _mm512_cvtepi32_ps(acc);
        
        // Scale Rehydration
        float s0 = decode_scale(w->scales[b]) * decode_scale(x->scales[b]);
        float s1 = decode_scale(w->scales[b+1]) * decode_scale(x->scales[b+1]);
        
        __m256 v_s0 = _mm256_set1_ps(s0);
        __m256 v_s1 = _mm256_set1_ps(s1);
        __m512 v_scales = _mm512_insertf32x8(_mm512_castps256_ps512(v_s0), v_s1, 1);
        
        f_acc = _mm512_mul_ps(f_acc, v_scales);
        global_sum += _mm512_reduce_add_ps(f_acc);
    }
    
    // Remainder 32-element block
    if (b < num_blocks) [[unlikely]] {
        // Guaranteed 32-byte alignment
        __m256i vW = _mm256_load_si256((const __m256i*)&w->data[b * 32]);
        __m256i vX = _mm256_load_si256((const __m256i*)&x->data[b * 32]);
        __m256i acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, vW);
        __m256 f_acc = _mm256_cvtepi32_ps(acc);
        float s = decode_scale(w->scales[b]) * decode_scale(x->scales[b]);
        f_acc = _mm256_mul_ps(f_acc, _mm256_set1_ps(s));
        
        __m128 v_low = _mm_add_ps(_mm256_castps256_ps128(f_acc), _mm256_extractf128_ps(f_acc, 1));
        v_low = _mm_hadd_ps(v_low, v_low);
        v_low = _mm_hadd_ps(v_low, v_low);
        global_sum += _mm_cvtss_f32(v_low);
    }
    
    return global_sum;
}

} // namespace nca::simd::avx512
