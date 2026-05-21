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
#include <algorithm>

namespace nca::layers {

void fused_gated_silu_quantize_scalar(const float* __restrict g, const float* __restrict u, nca::linalg::MXUINT8Tensor& out_q, size_t d) {
    for (size_t b = 0; b < d/32; ++b) {
        float ma = 0, h[32];
        for (int i=0; i<32; ++i) { auto x=g[b*32+i]; h[i]=(x/(1.f+std::exp(-x)))*u[b*32+i]; ma=std::max(ma,std::abs(h[i])); }
        out_q.scales[b] = nca::linalg::extract_e8m0(ma);
        auto inv = 1.f/nca::linalg::decode_e8m0_scale(out_q.scales[b]);
        for (int i=0; i<32; ++i) out_q.data[b*32+i] = (uint8_t)std::clamp(std::round(h[i]*inv)+128.f, 0.f, 255.f);
    }
}

#if defined(__AVX512F__) || defined(_MSC_VER)
void fused_gated_silu_quantize_avx512(const float* __restrict g, const float* __restrict u, nca::linalg::MXUINT8Tensor& out_q, size_t d) {
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
}

void fused_dual_gemv_avx512(const nca::linalg::MXINT8Tensor& Wg, const nca::linalg::MXINT8Tensor& Wu, const nca::linalg::MXUINT8Tensor& x, float* g, float* u, size_t rows, size_t cols) {
    const size_t bpr = cols / 32;
    for (size_t r = 0; r < rows; ++r) [[likely]] {
        nca::linalg::MXINT8Tensor v0, v1;
        auto p = [&](auto& v, const auto& W) { v.data=(int8_t*)W.data+r*cols; v.scales=W.scales+r*bpr; v.w_sums=W.w_sums+r*bpr; v.num_blocks=bpr; };
        p(v0, Wg); p(v1, Wu);
        nca::linalg::mx_dual_dot(v0, v1, x, g[r], u[r]);
        v0.data = v1.data = nullptr;
    }
}
#endif

void fused_gated_silu_quantize(const float* __restrict g, const float* __restrict u, nca::linalg::MXUINT8Tensor& out_q, size_t d) {
    if (simd::best_backend() == simd::Backend::AVX512) { fused_gated_silu_quantize_avx512(g, u, out_q, d); return; }
    fused_gated_silu_quantize_scalar(g, u, out_q, d);
}

void gated_mlp_step(const float* __restrict x, float* gb, float* ub, float* y, const nca::linalg::MXINT8Tensor& Wg, const nca::linalg::MXINT8Tensor& Wu, const nca::linalg::MXINT8Tensor& Wd, nca::linalg::MXUINT8Tensor& xq, nca::linalg::MXUINT8Tensor& hq, size_t dm, size_t di) {
    nca::linalg::mx_quantize_x(x, xq);
    if (simd::best_backend() == simd::Backend::AVX512) fused_dual_gemv_avx512(Wg, Wu, xq, gb, ub, di, dm);
    else { nca::linalg::mx_gemv(Wg, xq, gb, di, dm); nca::linalg::mx_gemv(Wu, xq, ub, di, dm); }
    fused_gated_silu_quantize(gb, ub, hq, di);
    nca::linalg::mx_gemv(Wd, hq, y, dm, di);
}

} // namespace nca::layers
