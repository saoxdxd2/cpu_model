// ============================================================================
// NCA -- Selective State-Space Model (SSM)
// core/layers/ssm.cpp
//
// CachePolicy analysis for SSM (D=8192, d_state=16):
//   h[]: 8192*16*4 = 512KB → DDR4_NT level
//   A[]: 8192*16*4 = 512KB → DDR4_NT level
//   Total working set: ~1.1MB → pure memory-bound
//
// Strategy: since h[] is read-modify-write (can't use NT stores),
// we maximize ILP (4x unroll) and use aggressive prefetching to hide
// DDR4 latency. The CachePolicy tells us prefetch_dist=8 cache lines.
// ============================================================================

#include "core/layers/ssm.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/cache_policy.hpp"
#include "core/simd/avx512_math.hpp"
#include <cmath>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace nca::layers {

// Compile-time analysis: 2 big arrays (h, A) of D*16 + 3 small (x, y, B, C)
// Total ≈ 1.1MB → DDR4_NT
using Policy = nca::simd::CachePolicy<2 * 8192 * 16 * 4 + 3 * 8192 * 4>;

void ssm_step_scalar(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C,
    const float* __restrict x,
    float* __restrict y,
    SSMConfig cfg
) {
    for (size_t d = 0; d < cfg.d_inner; ++d) {
        float x_d = x[d];
        float y_d = 0.0f;
        for (size_t s = 0; s < cfg.d_state; ++s) {
            size_t idx = d * cfg.d_state + s;
            float h_ds = A[idx] * h[idx] + B[s] * x_d;
            h[idx] = h_ds;
            y_d += h_ds * C[s];
        }
        y[d] = y_d;
    }
}

#if defined(__AVX512F__) || defined(_MSC_VER)
void ssm_step_avx512(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C,
    const float* __restrict x,
    float* __restrict y,
    size_t d_inner
) {
    // d_state == 16 → one __m512 register per state vector.
    // B, C are broadcast-hoisted — they stay in registers the entire kernel.
    __m512 v_B = _mm512_loadu_ps(B);
    __m512 v_C = _mm512_loadu_ps(C);

    // Policy::prefetch_dist = 8 cache lines = 512 bytes ahead
    constexpr size_t PF = Policy::prefetch_dist;

    // 4x unrolled: 4 channels = 4 cache lines of h[] + 4 of A[] per iteration
    size_t d = 0;
    for (; d + 4 <= d_inner; d += 4) [[likely]] {
        // Prefetch ahead: compile-time distance from CachePolicy
        if constexpr (PF > 0) {
            _mm_prefetch(reinterpret_cast<const char*>(&h[(d+PF) * 16]), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(&A[(d+PF) * 16]), _MM_HINT_T0);
        }

        __m512 v_x0 = _mm512_set1_ps(x[d]);
        __m512 v_x1 = _mm512_set1_ps(x[d+1]);
        __m512 v_x2 = _mm512_set1_ps(x[d+2]);
        __m512 v_x3 = _mm512_set1_ps(x[d+3]);

        __m512 v_h0 = _mm512_loadu_ps(&h[d * 16]);
        __m512 v_h1 = _mm512_loadu_ps(&h[(d+1) * 16]);
        __m512 v_h2 = _mm512_loadu_ps(&h[(d+2) * 16]);
        __m512 v_h3 = _mm512_loadu_ps(&h[(d+3) * 16]);

        // FMA: h = A*h + B*x (all in registers, zero extra loads for B)
        v_h0 = _mm512_fmadd_ps(_mm512_loadu_ps(&A[d * 16]),     v_h0, _mm512_mul_ps(v_B, v_x0));
        v_h1 = _mm512_fmadd_ps(_mm512_loadu_ps(&A[(d+1) * 16]), v_h1, _mm512_mul_ps(v_B, v_x1));
        v_h2 = _mm512_fmadd_ps(_mm512_loadu_ps(&A[(d+2) * 16]), v_h2, _mm512_mul_ps(v_B, v_x2));
        v_h3 = _mm512_fmadd_ps(_mm512_loadu_ps(&A[(d+3) * 16]), v_h3, _mm512_mul_ps(v_B, v_x3));

        // h[] is read-modify-write → regular stores (NT would evict before reduce)
        _mm512_storeu_ps(&h[d * 16],     v_h0);
        _mm512_storeu_ps(&h[(d+1) * 16], v_h1);
        _mm512_storeu_ps(&h[(d+2) * 16], v_h2);
        _mm512_storeu_ps(&h[(d+3) * 16], v_h3);

        // Output projection: y = reduce(C * h)
        y[d]   = _mm512_reduce_add_ps(_mm512_mul_ps(v_h0, v_C));
        y[d+1] = _mm512_reduce_add_ps(_mm512_mul_ps(v_h1, v_C));
        y[d+2] = _mm512_reduce_add_ps(_mm512_mul_ps(v_h2, v_C));
        y[d+3] = _mm512_reduce_add_ps(_mm512_mul_ps(v_h3, v_C));
    }
    // Scalar tail (at most 3 iterations — not worth masking)
    for (; d < d_inner; ++d) {
        __m512 v_x = _mm512_set1_ps(x[d]);
        __m512 v_h = _mm512_loadu_ps(&h[d * 16]);
        v_h = _mm512_fmadd_ps(_mm512_loadu_ps(&A[d * 16]), v_h, _mm512_mul_ps(v_B, v_x));
        _mm512_storeu_ps(&h[d * 16], v_h);
        y[d] = _mm512_reduce_add_ps(_mm512_mul_ps(v_h, v_C));
    }
}
#endif

void ssm_step(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C,
    const float* __restrict x,
    float* __restrict y,
    SSMConfig cfg
) {
    if (cfg.d_state == 16 && simd::best_backend() == simd::Backend::AVX512) [[likely]] {
#if defined(__AVX512F__) || defined(_MSC_VER)
        ssm_step_avx512(h, A, B, C, x, y, cfg.d_inner);
        return;
#endif
    }
    ssm_step_scalar(h, A, B, C, x, y, cfg);
}

void mx_fused_ssm_silu_quantize_scalar(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C,
    const float* __restrict x,
    nca::linalg::MXUINT8Tensor& y_q,
    SSMConfig cfg
) {
    size_t num_blocks = cfg.d_inner / 32;
    for (size_t b = 0; b < num_blocks; ++b) {
        float max_abs = 0.0f;
        float y_buf[32];
        for (size_t i = 0; i < 32; ++i) {
            size_t d = b * 32 + i;
            float x_d = x[d];
            float y_d = 0.0f;
            for (size_t s = 0; s < cfg.d_state; ++s) {
                size_t idx = d * cfg.d_state + s;
                float h_ds = A[idx] * h[idx] + B[s] * x_d;
                h[idx] = h_ds;
                y_d += h_ds * C[s];
            }
            // silu
            y_d = y_d / (1.0f + std::exp(-y_d));
            y_buf[i] = y_d;
            max_abs = std::max(max_abs, std::abs(y_d));
        }

        y_q.scales[b] = nca::linalg::extract_e8m0(max_abs);
        float scale = nca::linalg::decode_e8m0_scale(y_q.scales[b]);
        float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

        for (size_t i = 0; i < 32; ++i) {
            float val = std::round(y_buf[i] * inv_scale);
            y_q.data[b * 32 + i] = static_cast<uint8_t>(std::clamp(val + 128.0f, 0.0f, 255.0f));
        }
    }
}

#if defined(__AVX512F__) || defined(_MSC_VER)
void mx_fused_ssm_silu_quantize_avx512(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C,
    const float* __restrict x,
    nca::linalg::MXUINT8Tensor& y_q,
    size_t d_inner
) {
    __m512 v_B = _mm512_loadu_ps(B);
    __m512 v_C = _mm512_loadu_ps(C);
    constexpr size_t PF = Policy::prefetch_dist;

    size_t num_blocks = d_inner / 32;
    for (size_t b = 0; b < num_blocks; ++b) [[likely]] {
        float y_buf[32];
        for (size_t i = 0; i < 32; i += 4) {
            size_t d = b * 32 + i;
            if constexpr (PF > 0) {
                _mm_prefetch(reinterpret_cast<const char*>(&h[(d+PF) * 16]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&A[(d+PF) * 16]), _MM_HINT_T0);
            }

            __m512 v_x0 = _mm512_set1_ps(x[d]);
            __m512 v_x1 = _mm512_set1_ps(x[d+1]);
            __m512 v_x2 = _mm512_set1_ps(x[d+2]);
            __m512 v_x3 = _mm512_set1_ps(x[d+3]);

            __m512 v_h0 = _mm512_loadu_ps(&h[d * 16]);
            __m512 v_h1 = _mm512_loadu_ps(&h[(d+1) * 16]);
            __m512 v_h2 = _mm512_loadu_ps(&h[(d+2) * 16]);
            __m512 v_h3 = _mm512_loadu_ps(&h[(d+3) * 16]);

            v_h0 = _mm512_fmadd_ps(_mm512_loadu_ps(&A[d * 16]),     v_h0, _mm512_mul_ps(v_B, v_x0));
            v_h1 = _mm512_fmadd_ps(_mm512_loadu_ps(&A[(d+1) * 16]), v_h1, _mm512_mul_ps(v_B, v_x1));
            v_h2 = _mm512_fmadd_ps(_mm512_loadu_ps(&A[(d+2) * 16]), v_h2, _mm512_mul_ps(v_B, v_x2));
            v_h3 = _mm512_fmadd_ps(_mm512_loadu_ps(&A[(d+3) * 16]), v_h3, _mm512_mul_ps(v_B, v_x3));

            _mm512_storeu_ps(&h[d * 16],     v_h0);
            _mm512_storeu_ps(&h[(d+1) * 16], v_h1);
            _mm512_storeu_ps(&h[(d+2) * 16], v_h2);
            _mm512_storeu_ps(&h[(d+3) * 16], v_h3);

            y_buf[i]   = _mm512_reduce_add_ps(_mm512_mul_ps(v_h0, v_C));
            y_buf[i+1] = _mm512_reduce_add_ps(_mm512_mul_ps(v_h1, v_C));
            y_buf[i+2] = _mm512_reduce_add_ps(_mm512_mul_ps(v_h2, v_C));
            y_buf[i+3] = _mm512_reduce_add_ps(_mm512_mul_ps(v_h3, v_C));
        }

        __m512 v_y0 = _mm512_loadu_ps(&y_buf[0]);
        __m512 v_y1 = _mm512_loadu_ps(&y_buf[16]);

        v_y0 = nca::simd::avx512::silu_ps(v_y0);
        v_y1 = nca::simd::avx512::silu_ps(v_y1);

        __m512 v_abs0 = _mm512_abs_ps(v_y0);
        __m512 v_abs1 = _mm512_abs_ps(v_y1);
        float max_val = _mm512_reduce_max_ps(_mm512_max_ps(v_abs0, v_abs1));

        y_q.scales[b] = nca::linalg::extract_e8m0(max_val);
        float scale = nca::linalg::decode_e8m0_scale(y_q.scales[b]);
        float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        __m512 v_inv = _mm512_set1_ps(inv_scale);

        __m512 v_q0 = _mm512_mul_ps(v_y0, v_inv);
        __m512 v_q1 = _mm512_mul_ps(v_y1, v_inv);

        __m512i v_i0 = _mm512_cvtps_epi32(v_q0);
        __m512i v_i1 = _mm512_cvtps_epi32(v_q1);

        // Offset by 128 and clamp to [0, 255] — branchless via min/max
        __m512i v_128 = _mm512_set1_epi32(128);
        v_i0 = _mm512_add_epi32(v_i0, v_128);
        v_i1 = _mm512_add_epi32(v_i1, v_128);
        v_i0 = _mm512_min_epi32(_mm512_max_epi32(v_i0, _mm512_setzero_si512()), _mm512_set1_epi32(255));
        v_i1 = _mm512_min_epi32(_mm512_max_epi32(v_i1, _mm512_setzero_si512()), _mm512_set1_epi32(255));

        _mm_storeu_si128(reinterpret_cast<__m128i*>(&y_q.data[b * 32]),      _mm512_cvtepi32_epi8(v_i0));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(&y_q.data[b * 32 + 16]), _mm512_cvtepi32_epi8(v_i1));
    }
}
#endif

void mx_fused_ssm_silu_quantize_step(
    float* __restrict h,
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict C,
    const float* __restrict x,
    nca::linalg::MXUINT8Tensor& y_q,
    SSMConfig cfg
) {
    if (cfg.d_state == 16 && simd::best_backend() == simd::Backend::AVX512) [[likely]] {
#if defined(__AVX512F__) || defined(_MSC_VER)
        mx_fused_ssm_silu_quantize_avx512(h, A, B, C, x, y_q, cfg.d_inner);
        return;
#endif
    }
    mx_fused_ssm_silu_quantize_scalar(h, A, B, C, x, y_q, cfg);
}

} // namespace nca::layers
