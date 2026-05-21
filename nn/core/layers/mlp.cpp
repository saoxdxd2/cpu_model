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

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace nca::layers {

void fused_gated_silu_quantize_scalar(
    const float* __restrict gate,
    const float* __restrict up,
    nca::linalg::MXUINT8Tensor& out_q,
    size_t d_inner
) {
    auto num_blocks = d_inner / 32;
    for (size_t b = 0; b < num_blocks; ++b) {
        float max_abs = 0.0f;
        float hidden[32];
        for (size_t i = 0; i < 32; ++i) {
            auto g = gate[b * 32 + i];
            auto u = up[b * 32 + i];
            auto s = g / (1.0f + std::exp(-g));
            auto h = s * u;
            hidden[i] = h;
            max_abs = std::max(max_abs, std::abs(h));
        }

        out_q.scales[b] = nca::linalg::extract_e8m0(max_abs);
        auto scale = nca::linalg::decode_e8m0_scale(out_q.scales[b]);
        auto inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

        for (size_t i = 0; i < 32; ++i) {
            auto val = std::round(hidden[i] * inv_scale);
            out_q.data[b * 32 + i] = static_cast<uint8_t>(std::clamp(val + 128.0f, 0.0f, 255.0f));
        }
    }
}

#if defined(__AVX512F__) || defined(_MSC_VER)
void fused_gated_silu_quantize_avx512(
    const float* __restrict gate,
    const float* __restrict up,
    nca::linalg::MXUINT8Tensor& out_q,
    size_t d_inner
) {
    auto num_blocks = d_inner / 32;
    
    // 2x Unrolled Block Processing (64 elements per iter)
    size_t b = 0;
    for (; b + 1 < num_blocks; b += 2) [[likely]] {
        auto g_ptr = &gate[b * 32];
        auto u_ptr = &up[b * 32];
        
        auto v_g0 = _mm512_loadu_ps(g_ptr);
        auto v_g1 = _mm512_loadu_ps(g_ptr + 16);
        auto v_g2 = _mm512_loadu_ps(g_ptr + 32);
        auto v_g3 = _mm512_loadu_ps(g_ptr + 48);

        auto v_u0 = _mm512_loadu_ps(u_ptr);
        auto v_u1 = _mm512_loadu_ps(u_ptr + 16);
        auto v_u2 = _mm512_loadu_ps(u_ptr + 32);
        auto v_u3 = _mm512_loadu_ps(u_ptr + 48);
        
        auto v_h0 = _mm512_mul_ps(nca::simd::avx512::silu_ps(v_g0), v_u0);
        auto v_h1 = _mm512_mul_ps(nca::simd::avx512::silu_ps(v_g1), v_u1);
        auto v_h2 = _mm512_mul_ps(nca::simd::avx512::silu_ps(v_g2), v_u2);
        auto v_h3 = _mm512_mul_ps(nca::simd::avx512::silu_ps(v_g3), v_u3);
        
        auto quantize_block = [&](auto v0, auto v1, size_t block_idx) {
            auto v_max = _mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1));
            auto max_val = _mm512_reduce_max_ps(v_max);
            out_q.scales[block_idx] = nca::linalg::extract_e8m0(max_val);
            auto inv_scale = 1.0f / nca::linalg::decode_e8m0_scale(out_q.scales[block_idx]);
            auto v_inv = _mm512_set1_ps(inv_scale);
            auto v_i0 = _mm512_cvtps_epi32(_mm512_mul_ps(v0, v_inv));
            auto v_i1 = _mm512_cvtps_epi32(_mm512_mul_ps(v1, v_inv));
            auto v_128 = _mm512_set1_epi32(128);
            v_i0 = _mm512_add_epi32(v_i0, v_128);
            v_i1 = _mm512_add_epi32(v_i1, v_128);
            v_i0 = _mm512_min_epi32(_mm512_max_epi32(v_i0, _mm512_setzero_si512()), _mm512_set1_epi32(255));
            v_i1 = _mm512_min_epi32(_mm512_max_epi32(v_i1, _mm512_setzero_si512()), _mm512_set1_epi32(255));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&out_q.data[block_idx * 32]), _mm512_cvtepi32_epi8(v_i0));
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&out_q.data[block_idx * 32 + 16]), _mm512_cvtepi32_epi8(v_i1));
        };

        quantize_block(v_h0, v_h1, b);
        quantize_block(v_h2, v_h3, b + 1);
    }
    for (; b < num_blocks; ++b) {
        auto g_ptr = &gate[b * 32];
        auto u_ptr = &up[b * 32];
        auto v_g0 = _mm512_loadu_ps(g_ptr);
        auto v_g1 = _mm512_loadu_ps(g_ptr + 16);
        auto v_u0 = _mm512_loadu_ps(u_ptr);
        auto v_u1 = _mm512_loadu_ps(u_ptr + 16);
        auto v_h0 = _mm512_mul_ps(nca::simd::avx512::silu_ps(v_g0), v_u0);
        auto v_h1 = _mm512_mul_ps(nca::simd::avx512::silu_ps(v_g1), v_u1);
        
        auto v_max = _mm512_max_ps(_mm512_abs_ps(v_h0), _mm512_abs_ps(v_h1));
        auto max_val = _mm512_reduce_max_ps(v_max);
        out_q.scales[b] = nca::linalg::extract_e8m0(max_val);
        auto inv_scale = 1.0f / nca::linalg::decode_e8m0_scale(out_q.scales[b]);
        auto v_inv = _mm512_set1_ps(inv_scale);
        auto v_i0 = _mm512_cvtps_epi32(_mm512_mul_ps(v_h0, v_inv));
        auto v_i1 = _mm512_cvtps_epi32(_mm512_mul_ps(v_h1, v_inv));
        auto v_128 = _mm512_set1_epi32(128);
        v_i0 = _mm512_add_epi32(v_i0, v_128);
        v_i1 = _mm512_add_epi32(v_i1, v_128);
        v_i0 = _mm512_min_epi32(_mm512_max_epi32(v_i0, _mm512_setzero_si512()), _mm512_set1_epi32(255));
        v_i1 = _mm512_min_epi32(_mm512_max_epi32(v_i1, _mm512_setzero_si512()), _mm512_set1_epi32(255));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out_q.data[b * 32]), _mm512_cvtepi32_epi8(v_i0));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out_q.data[b * 32 + 16]), _mm512_cvtepi32_epi8(v_i1));
    }
}
#endif

void fused_gated_silu_quantize(
    const float* __restrict gate,
    const float* __restrict up,
    nca::linalg::MXUINT8Tensor& out_q,
    size_t d_inner
) {
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
#if defined(__AVX512F__) || defined(_MSC_VER)
        fused_gated_silu_quantize_avx512(gate, up, out_q, d_inner);
        return;
#endif
    }
    fused_gated_silu_quantize_scalar(gate, up, out_q, d_inner);
}

#if defined(__AVX512F__) || defined(_MSC_VER)
void fused_dual_gemv_avx512(
    const nca::linalg::MXINT8Tensor& W_gate,
    const nca::linalg::MXINT8Tensor& W_up,
    const nca::linalg::MXUINT8Tensor& x_q,
    float* __restrict gate,
    float* __restrict up,
    size_t rows,
    size_t cols
) {
    const auto blocks_per_row = cols / 32;
    
    // 2x Unrolled Pair Processing (process 2 rows of BOTH matrices per iter)
    size_t r = 0;
    for (; r + 1 < rows; r += 2) [[likely]] {
        auto prep_view = [&](const nca::linalg::MXINT8Tensor& W, size_t row_idx) {
            nca::linalg::MXINT8Tensor v;
            v.data    = const_cast<int8_t*>(W.data + row_idx * cols);
            v.scales  = const_cast<uint8_t*>(W.scales + row_idx * blocks_per_row);
            v.w_sums  = const_cast<int32_t*>(W.w_sums + row_idx * blocks_per_row);
            v.num_blocks = blocks_per_row;
            return v;
        };

        auto v_g0 = prep_view(W_gate, r);
        auto v_u0 = prep_view(W_up, r);
        auto v_g1 = prep_view(W_gate, r + 1);
        auto v_u1 = prep_view(W_up, r + 1);

        nca::linalg::mx_dual_dot(v_g0, v_u0, x_q, gate[r], up[r]);
        nca::linalg::mx_dual_dot(v_g1, v_u1, x_q, gate[r + 1], up[r + 1]);

        v_g0.data = v_u0.data = v_g1.data = v_u1.data = nullptr;
    }
    for (; r < rows; ++r) {
        auto prep_view = [&](const nca::linalg::MXINT8Tensor& W, size_t row_idx) {
            nca::linalg::MXINT8Tensor v;
            v.data    = const_cast<int8_t*>(W.data + row_idx * cols);
            v.scales  = const_cast<uint8_t*>(W.scales + row_idx * blocks_per_row);
            v.w_sums  = const_cast<int32_t*>(W.w_sums + row_idx * blocks_per_row);
            v.num_blocks = blocks_per_row;
            return v;
        };
        auto v_g = prep_view(W_gate, r);
        auto v_u = prep_view(W_up, r);
        nca::linalg::mx_dual_dot(v_g, v_u, x_q, gate[r], up[r]);
        v_g.data = v_u.data = nullptr;
    }
}
#endif

void gated_mlp_step(
    const float* __restrict x,
    float* __restrict gate_buf,
    float* __restrict up_buf,
    float* __restrict y,
    const nca::linalg::MXINT8Tensor& W_gate,
    const nca::linalg::MXINT8Tensor& W_up,
    const nca::linalg::MXINT8Tensor& W_down,
    nca::linalg::MXUINT8Tensor& x_q,
    nca::linalg::MXUINT8Tensor& hidden_q,
    size_t d_model,
    size_t d_inner
) {
    nca::linalg::mx_quantize_x(x, x_q);
    
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
#if defined(__AVX512F__) || defined(_MSC_VER)
        fused_dual_gemv_avx512(W_gate, W_up, x_q, gate_buf, up_buf, d_inner, d_model);
#endif
    } else {
        nca::linalg::mx_gemv(W_gate, x_q, gate_buf, d_inner, d_model);
        nca::linalg::mx_gemv(W_up, x_q, up_buf, d_inner, d_model);
    }
    
    fused_gated_silu_quantize(gate_buf, up_buf, hidden_q, d_inner);
    nca::linalg::mx_gemv(W_down, hidden_q, y, d_model, d_inner);
}

} // namespace nca::layers
