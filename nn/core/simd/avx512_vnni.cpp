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

// ── RANK-16 FUSION ──────────────────────────────────────────────────────────
void rank16_vnni_dot(const nca::linalg::MXINT8Tensor* __restrict ws, const nca::linalg::MXUINT8Tensor* __restrict x, float* __restrict out) {
    const size_t num_blocks = x->num_blocks;
    __m512 v_acc_all[16];
    for(int i=0; i<16; ++i) v_acc_all[i] = _mm512_setzero_ps();
    for (size_t b = 0; b < num_blocks; ++b) [[likely]] {
        __m256i vX = _mm256_loadu_si256((const __m256i*)&x->data[b * 32]);
        uint8_t ex = x->scales[b];
        alignas(64) uint32_t ews[16];
        for(int i=0; i<16; ++i) ews[i] = (uint32_t)ws[i].scales[b];
        __m512 vS = scale_fusion_ps(_mm512_load_si512(ews), ex);
        alignas(64) float s_buf[16]; _mm512_store_ps(s_buf, vS);

        for (int i = 0; i < 16; ++i) {
            __m256i i_acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, _mm256_loadu_si256((const __m256i*)&ws[i].data[b*32]));
            __m128i c = _mm_add_epi32(_mm256_castsi256_si128(i_acc), _mm256_extracti128_si256(i_acc, 1));
            c = _mm_add_epi32(c, _mm_srli_si128(c, 8)); c = _mm_add_epi32(c, _mm_srli_si128(c, 4));
            v_acc_all[i] = _mm512_fmadd_ps(_mm512_set1_ps((float)_mm_cvtsi128_si32(c) - 128.f * ws[i].w_sums[b]), _mm512_set1_ps(s_buf[i]), v_acc_all[i]);
        }
    }
    for(int i=0; i<16; ++i) out[i] = _mm512_reduce_add_ps(v_acc_all[i]);
}

void rank16_vnni_dot_ptrs(const nca::linalg::MXINT8Tensor** __restrict ws, const nca::linalg::MXUINT8Tensor* __restrict x, float* __restrict out) {
    const size_t num_blocks = x->num_blocks;
    __m512 v_acc_all[16];
    for(int i=0; i<16; ++i) v_acc_all[i] = _mm512_setzero_ps();
    for (size_t b = 0; b < num_blocks; ++b) [[likely]] {
        __m256i vX = _mm256_loadu_si256((const __m256i*)&x->data[b * 32]);
        uint8_t ex = x->scales[b];
        alignas(64) uint32_t ews[16];
        for(int i=0; i<16; ++i) ews[i] = (uint32_t)ws[i]->scales[b];
        __m512 vS = scale_fusion_ps(_mm512_load_si512(ews), ex);
        alignas(64) float s_buf[16]; _mm512_store_ps(s_buf, vS);

        for (int i = 0; i < 16; ++i) {
            __m256i i_acc = _mm256_dpbusd_epi32(_mm256_setzero_si256(), vX, _mm256_loadu_si256((const __m256i*)&ws[i]->data[b*32]));
            __m128i c = _mm_add_epi32(_mm256_castsi256_si128(i_acc), _mm256_extracti128_si256(i_acc, 1));
            c = _mm_add_epi32(c, _mm_srli_si128(c, 8)); c = _mm_add_epi32(c, _mm_srli_si128(c, 4));
            v_acc_all[i] = _mm512_fmadd_ps(_mm512_set1_ps((float)_mm_cvtsi128_si32(c) - 128.f * ws[i]->w_sums[b]), _mm512_set1_ps(s_buf[i]), v_acc_all[i]);
        }
    }
    for(int i=0; i<16; ++i) out[i] = _mm512_reduce_add_ps(v_acc_all[i]);
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

} // namespace nca::simd::avx512
