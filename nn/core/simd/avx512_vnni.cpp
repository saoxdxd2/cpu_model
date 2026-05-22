// ============================================================================
// NCA — AVX-512 VNNI Kernels
// core/simd/avx512_vnni.cpp
// ============================================================================

#include "core/simd/avx512_vnni.hpp"
#include "core/simd/avx512_math.hpp"
#include <immintrin.h>
#include <algorithm>

namespace nca::simd::avx512 {

// ── LOG-SPACE SCALING (Target 3) ───────────────────────────────────────────
inline __m512 scale_fusion_ps(__m512i vEw, uint8_t ex) {
    if (ex == 0) return _mm512_setzero_ps();
    __m512i vTotalExp = _mm512_sub_epi32(_mm512_add_epi32(vEw, _mm512_set1_epi32((int)ex)), _mm512_set1_epi32(254));
    return _mm512_scalef_ps(_mm512_set1_ps(1.0f), _mm512_cvtepi32_ps(vTotalExp));
}

// ── QUAD-CHAIN ILP DOT PRODUCT ──────────────────────────────────────────────
float vnni_dot(const nca::linalg::MXINT8Tensor* __restrict w, const nca::linalg::MXUINT8Tensor* __restrict x) {
    const size_t num_blocks = std::min(w->num_blocks, x->num_blocks);
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    
    size_t b = 0;
    for (; b + 3 < num_blocks; b += 4) [[likely]] {
        auto step = [&](size_t bi, __m512& a) {
            __m256i i = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_loadu_si256((const __m256i*)&x->data[bi*32]), _mm256_loadu_si256((const __m256i*)&w->data[bi*32]));
            __m128i c = _mm_add_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1));
            c = _mm_add_epi32(c, _mm_srli_si128(c, 8)); c = _mm_add_epi32(c, _mm_srli_si128(c, 4));
            float s = nca::linalg::decode_e8m0_scale(w->scales[bi]) * nca::linalg::decode_e8m0_scale(x->scales[bi]);
            a = _mm512_fmadd_ps(_mm512_set1_ps((float)_mm_cvtsi128_si32(c) - 128.f * w->w_sums[bi]), _mm512_set1_ps(s), a);
        };
        step(b, acc0); step(b+1, acc1); step(b+2, acc2); step(b+3, acc3);
    }
    for (; b < num_blocks; ++b) {
        __m256i i = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_loadu_si256((const __m256i*)&x->data[b*32]), _mm256_loadu_si256((const __m256i*)&w->data[b*32]));
        __m128i c = _mm_add_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1));
        c = _mm_add_epi32(c, _mm_srli_si128(c, 8)); c = _mm_add_epi32(c, _mm_srli_si128(c, 4));
        float s = nca::linalg::decode_e8m0_scale(w->scales[b]) * nca::linalg::decode_e8m0_scale(x->scales[b]);
        acc0 = _mm512_fmadd_ps(_mm512_set1_ps((float)_mm_cvtsi128_si32(c) - 128.f * w->w_sums[b]), _mm512_set1_ps(s), acc0);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
}

// ── RANK-16 FUSION (Absolute Saturation) ───────────────────────────────────
void rank16_vnni_dot(const nca::linalg::MXINT8Tensor* ws, const nca::linalg::MXUINT8Tensor* x, float* out) {
    const size_t num_blocks = x->num_blocks;
    __m512 v_acc = _mm512_setzero_ps();
    
    for (size_t b = 0; b < num_blocks; ++b) [[likely]] {
        __m256i vX = _mm256_loadu_si256((const __m256i*)&x->data[b * 32]);
        uint8_t ex = x->scales[b];
        
        alignas(64) uint32_t ews[16];
        alignas(64) int32_t dots[16];
        alignas(64) int32_t w_sums[16];
        
        for (int i = 0; i < 16; ++i) {
            ews[i] = (uint32_t)ws[i].scales[b];
            w_sums[i] = ws[i].w_sums[b];
            
            __m256i i_acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, _mm256_loadu_si256((const __m256i*)&ws[i].data[b*32]));
            __m128i c = _mm_add_epi32(_mm256_castsi256_si128(i_acc), _mm256_extracti128_si256(i_acc, 1));
            c = _mm_add_epi32(c, _mm_srli_si128(c, 8)); c = _mm_add_epi32(c, _mm_srli_si128(c, 4));
            dots[i] = _mm_cvtsi128_si32(c);
        }
        
        __m512 vS = scale_fusion_ps(_mm512_load_si512(ews), ex);
        __m512 vH = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_load_si512(dots)), 
                                  _mm512_mul_ps(_mm512_set1_ps(128.0f), _mm512_cvtepi32_ps(_mm512_load_si512(w_sums))));
        
        v_acc = _mm512_fmadd_ps(vH, vS, v_acc);
    }
    _mm512_storeu_ps(out, v_acc);
}

void rank16_vnni_dot_ptrs(const nca::linalg::MXINT8Tensor** ws, const nca::linalg::MXUINT8Tensor* x, float* out) {
    const size_t num_blocks = x->num_blocks;
    __m512 v_acc = _mm512_setzero_ps();
    
    for (size_t b = 0; b < num_blocks; ++b) [[likely]] {
        __m256i vX = _mm256_loadu_si256((const __m256i*)&x->data[b * 32]);
        uint8_t ex = x->scales[b];
        
        alignas(64) uint32_t ews[16];
        alignas(64) int32_t dots[16];
        alignas(64) int32_t w_sums[16];
        
        for (int i = 0; i < 16; ++i) {
            ews[i] = (uint32_t)ws[i]->scales[b];
            w_sums[i] = ws[i]->w_sums[b];
            
            __m256i i_acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, _mm256_loadu_si256((const __m256i*)&ws[i]->data[b*32]));
            __m128i c = _mm_add_epi32(_mm256_castsi256_si128(i_acc), _mm256_extracti128_si256(i_acc, 1));
            c = _mm_add_epi32(c, _mm_srli_si128(c, 8)); c = _mm_add_epi32(c, _mm_srli_si128(c, 4));
            dots[i] = _mm_cvtsi128_si32(c);
        }
        
        __m512 vS = scale_fusion_ps(_mm512_load_si512(ews), ex);
        __m512 vH = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_load_si512(dots)), 
                                  _mm512_mul_ps(_mm512_set1_ps(128.0f), _mm512_cvtepi32_ps(_mm512_load_si512(w_sums))));
        
        v_acc = _mm512_fmadd_ps(vH, vS, v_acc);
    }
    _mm512_storeu_ps(out, v_acc);
}

void mx_quantize_x(const float* __restrict in, nca::linalg::MXUINT8Tensor* __restrict out) {
    for (size_t b = 0; b < out->num_blocks; ++b) [[likely]] {
        __m512 v0 = _mm512_loadu_ps(&in[b*32]), v1 = _mm512_loadu_ps(&in[b*32 + 16]);
        float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        out->scales[b] = nca::linalg::extract_e8m0(max_abs);
        __m512 v_inv = _mm512_set1_ps(1.0f / nca::linalg::decode_e8m0_scale(out->scales[b]));
        auto q = [&](auto v, int off) {
            auto vi = _mm512_min_epi32(_mm512_max_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v, v_inv)), _mm512_set1_epi32(128)), _mm512_setzero_si512()), _mm512_set1_epi32(255));
            _mm_storeu_si128((__m128i*)&out->data[b*32+off], _mm512_cvtepi32_epi8(vi));
        };
        q(v0, 0); q(v1, 16);
    }
}

void mx_fused_silu_quantize_x(const float* __restrict in, nca::linalg::MXUINT8Tensor* __restrict out) {
    for (size_t b = 0; b < out->num_blocks; ++b) [[likely]] {
        auto v0 = nca::simd::avx512::silu_ps(_mm512_loadu_ps(&in[b*32])), v1 = nca::simd::avx512::silu_ps(_mm512_loadu_ps(&in[b*32+16]));
        float max_abs = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1)));
        out->scales[b] = nca::linalg::extract_e8m0(max_abs);
        __m512 v_inv = _mm512_set1_ps(1.0f / nca::linalg::decode_e8m0_scale(out->scales[b]));
        auto q = [&](auto v, int off) {
            auto vi = _mm512_min_epi32(_mm512_max_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v, v_inv)), _mm512_set1_epi32(128)), _mm512_setzero_si512()), _mm512_set1_epi32(255));
            _mm_storeu_si128((__m128i*)&out->data[b*32+off], _mm512_cvtepi32_epi8(vi));
        };
        q(v0, 0); q(v1, 16);
    }
}

float mx_compute_activation_norm(const nca::linalg::MXUINT8Tensor& x_q) {
    __m512 v_sum_sq = _mm512_setzero_ps();
    for (size_t b = 0; b < x_q.num_blocks; ++b) {
        __m512 v_scale = _mm512_set1_ps(nca::linalg::decode_e8m0_scale(x_q.scales[b]));
        __m256i v_data = _mm256_loadu_si256((const __m256i*)&x_q.data[b * 32]);
        
        __m512i v_i32_lo = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(v_data));
        __m512i v_i32_hi = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(v_data, 1));
        
        __m512 v_f32_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(v_i32_lo, _mm512_set1_epi32(128))), v_scale);
        __m512 v_f32_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(v_i32_hi, _mm512_set1_epi32(128))), v_scale);
        
        v_sum_sq = _mm512_fmadd_ps(v_f32_lo, v_f32_lo, v_sum_sq);
        v_sum_sq = _mm512_fmadd_ps(v_f32_hi, v_f32_hi, v_sum_sq);
    }
    return _mm512_reduce_add_ps(v_sum_sq) + 1e-4f;
}

void mx_update_gaussian_moment(
    nca::linalg::MXINT8Tensor& w, 
    const nca::linalg::MXUINT8Tensor& x_q, 
    float error, 
    float lr, 
    float precomputed_norm
) {
    const size_t bpr = x_q.num_blocks;
    float norm = (precomputed_norm > 0) ? precomputed_norm : nca::simd::avx512::mx_compute_activation_norm(x_q);
    
    float current_total_w = 0;
    for (size_t b = 0; b < bpr; ++b) current_total_w += nca::linalg::decode_e8m0_scale(w.scales[b]);
    if (current_total_w > 5.0f && lr > 0) lr *= 0.05f; 

    __m512 v_lr_err_norm = _mm512_set1_ps((lr * error) / norm);

    for (size_t b = 0; b < bpr; ++b) {
        float sw = nca::linalg::decode_e8m0_scale(w.scales[b]);
        __m512 v_sw = _mm512_set1_ps(sw);
        __m512 v_sx = _mm512_set1_ps(nca::linalg::decode_e8m0_scale(x_q.scales[b]));
        __m512 v_grad_scale = _mm512_mul_ps(v_lr_err_norm, v_sx);
        __m512 v_inv_sw = _mm512_set1_ps(1.0f / (sw + 1e-6f));

        __m256i v_w_data = _mm256_loadu_si256((const __m256i*)&w.data[b * 32]);
        __m256i v_x_data = _mm256_loadu_si256((const __m256i*)&x_q.data[b * 32]);

        auto process_half = [&](__m128i w128, __m128i x128, int offset) {
            __m512i v_wi = _mm512_cvtepi8_epi32(w128);
            __m512i v_xi = _mm512_sub_epi32(_mm512_cvtepu8_epi32(x128), _mm512_set1_epi32(128));
            __m512 v_wf = _mm512_mul_ps(_mm512_cvtepi32_ps(v_wi), v_sw);
            __m512 v_new_w = _mm512_fmadd_ps(v_grad_scale, _mm512_cvtepi32_ps(v_xi), v_wf);
            
            __m512i qi = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(v_new_w, v_inv_sw), _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
            qi = _mm512_min_epi32(_mm512_max_epi32(qi, _mm512_set1_epi32(-127)), _mm512_set1_epi32(127));
            _mm_storeu_si128((__m128i*)&w.data[b*32 + offset], _mm512_cvtepi32_epi8(qi));
            return _mm512_reduce_add_epi32(qi);
        };

        w.w_sums[b] = process_half(_mm256_castsi256_si128(v_w_data), _mm256_castsi256_si128(v_x_data), 0) + 
                      process_half(_mm256_extracti128_si256(v_w_data, 1), _mm256_extracti128_si256(v_x_data, 1), 16);
    }
}

} // namespace nca::simd::avx512
