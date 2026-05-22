// ============================================================================
// NCA -- Gated MLP Layer (Phase 7)
// core/layers/mlp.cpp
// ============================================================================

#include "core/layers/mlp.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/cache_policy.hpp"
#include "core/simd/avx512_math.hpp"
#include "core/simd/avx512_vnni.hpp"
#include <cmath>

namespace nca::layers {

void fused_gated_silu_quantize(const float* __restrict g, const float* __restrict u, nca::linalg::MXUINT8Tensor& out_q, size_t d) {
#if defined(__AVX512F__) || defined(_MSC_VER)
    for (size_t b = 0; b < d/32; ++b) [[likely]] {
        auto v_h0 = _mm512_mul_ps(nca::simd::avx512::silu_ps(_mm512_loadu_ps(g+b*32)), _mm512_loadu_ps(u+b*32));
        auto v_h1 = _mm512_mul_ps(nca::simd::avx512::silu_ps(_mm512_loadu_ps(g+b*32+16)), _mm512_loadu_ps(u+b*32+16));
        auto v_ma = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(v_h0), _mm512_abs_ps(v_h1)));
        out_q.scales[b] = nca::linalg::extract_e8m0(v_ma);
        auto v_inv = _mm512_set1_ps(1.f/nca::linalg::decode_e8m0_scale(out_q.scales[b]));
        auto v_128 = _mm512_set1_epi32(128);
        auto q = [&](auto v, int off) {
            auto vi = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(v, v_inv)), v_128);
            vi = _mm512_min_epi32(_mm512_max_epi32(vi, _mm512_setzero_si512()), _mm512_set1_epi32(255));
            _mm_storeu_si128((__m128i*)&out_q.data[b*32+off], _mm512_cvtepi32_epi8(vi));
        };
        q(v_h0, 0); q(v_h1, 16);
    }
#else
    for (size_t b = 0; b < d/32; ++b) {
        float ma = 0.0f;
        float h[32];
        for (int i=0; i<32; ++i) { 
            float x = g[b*32+i];
            float sx = 1.0f / (1.0f + std::exp(-x));
            float val = (x * sx) * u[b*32+i];
            h[i] = val;
            float abs_val = val < 0.0f ? -val : val;
            ma = abs_val > ma ? abs_val : ma;
        }
        out_q.scales[b] = nca::linalg::extract_e8m0(ma);
        float inv = 1.0f / nca::linalg::decode_e8m0_scale(out_q.scales[b]);
        for (int i=0; i<32; ++i) {
            float scaled = h[i] * inv + 128.0f;
            int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
            int32_t c1 = r < 0 ? 0 : r;
            int32_t c2 = c1 > 255 ? 255 : c1;
            out_q.data[b*32+i] = (uint8_t)c2;
        }
    }
#endif
}

void gated_mlp_step(const float* __restrict x, float* gb, float* ub, float* y, const nca::linalg::MXINT8Tensor& Wg, const nca::linalg::MXINT8Tensor& Wu, const nca::linalg::MXINT8Tensor& Wd, nca::linalg::MXUINT8Tensor& xq, nca::linalg::MXUINT8Tensor& hq, size_t dm, size_t di) {
    nca::linalg::mx_quantize_x(x, xq);
    
    // ── GEOMETRIC MANDATE: Rank-16 Block Processing ────────────────────────
    // Unconditionally use mx_gemv to ensure xq is reused 16 times in L1.
    nca::linalg::mx_gemv(Wg, xq, gb, di, dm);
    nca::linalg::mx_gemv(Wu, xq, ub, di, dm);

    fused_gated_silu_quantize(gb, ub, hq, di);
    nca::linalg::mx_gemv(Wd, hq, y, dm, di);
}

} // namespace nca::layers

