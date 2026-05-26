// ============================================================================
// NCA -- Sparse Local Attention (SLA) Layer
// core/layers/sla.cpp
//
// CachePolicy analysis for SLA (d_head=128, W=256):
//   q:       128*4 = 512 bytes  (fits one cache line set)
//   scores:  256*4 = 1KB
//   k_cache: 256*128*4 = 128KB  (fits L2)
//   v_cache: 256*128*4 = 128KB  (fits L2)
//   Total: ~257KB → L2_STREAM
//
// Phase 1: QK dot products → 256 scores (compute-bound, FMA-dominated)
// Phase 2: Online softmax   → branchless max + exp + normalize
// Phase 3: Score * V accumulation → weighted sum (memory-bound on V cache)
// ============================================================================

#include "core/layers/sla.hpp"
#include "core/simd/dispatch.hpp"
#include "core/simd/cache_policy.hpp"
#include "core/simd/avx512_math.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <cmath>

namespace nca::layers {

// Compile-time: k_cache + v_cache = 2 * 256 * 128 * 4 = 256KB → L2_STREAM
using Policy = nca::simd::CachePolicy<2 * 256 * 128 * 4>;

void sla_step_scalar(
    const float* __restrict q,
    const float* __restrict k_cache,
    const float* __restrict v_cache,
    float* __restrict out,
    float* __restrict scores,
    SLAConfig cfg
) {
    float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(cfg.d_head));

    // Phase 1: QK dot products
    for (size_t j = 0; j < cfg.kv_len; ++j) {
        float dot = 0.0f;
        for (size_t k = 0; k < cfg.d_head; ++k)
            dot += q[k] * k_cache[j * cfg.d_head + k];
        scores[j] = dot * inv_sqrt;
    }

    // Phase 2: Softmax
    float max_s = scores[0];
    for (size_t j = 1; j < cfg.kv_len; ++j)
        max_s = std::max(max_s, scores[j]);

    float sum_exp = 0.0f;
    for (size_t j = 0; j < cfg.kv_len; ++j) {
        scores[j] = std::exp(scores[j] - max_s);
        sum_exp += scores[j];
    }
    float inv_sum = 1.0f / sum_exp;
    for (size_t j = 0; j < cfg.kv_len; ++j)
        scores[j] *= inv_sum;

    // Phase 3: Weighted V sum
    for (size_t k = 0; k < cfg.d_head; ++k) out[k] = 0.0f;
    for (size_t j = 0; j < cfg.kv_len; ++j) {
        float s = scores[j];
        for (size_t k = 0; k < cfg.d_head; ++k)
            out[k] += s * v_cache[j * cfg.d_head + k];
    }
}

#if defined(__AVX512F__) || defined(_MSC_VER)
void sla_step_avx512(
    const float* __restrict q,
    const float* __restrict k_cache,
    const float* __restrict v_cache,
    float* __restrict out,
    float* __restrict scores,
    SLAConfig cfg
) {
    const size_t D = cfg.d_head;
    const size_t W = cfg.kv_len;
    float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(D));
    __m512 v_scale = _mm512_set1_ps(inv_sqrt);

    constexpr size_t PF = Policy::prefetch_dist;

    // ── Phase 1: QK dot products ─────────────────────────────────────────
    // For each key j in [0, W): score[j] = (q · k[j]) / sqrt(d)
    // d_head=128 = 8 x __m512 registers. Fully unrollable.
    for (size_t j = 0; j < W; ++j) [[likely]] {
        const float* k_row = &k_cache[j * D];

        if constexpr (PF > 0) {
            if (j + PF < W)
                _mm_prefetch(reinterpret_cast<const char*>(&k_cache[(j + PF) * D]), _MM_HINT_T0);
        }

        // [ILP] 4-way accumulator — saturates FMA ports for d_head=128
        __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();

        size_t k = 0;
        for (; k + 64 <= D; k += 64) {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(&q[k]),      _mm512_loadu_ps(&k_row[k]),      acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(&q[k + 16]), _mm512_loadu_ps(&k_row[k + 16]), acc1);
            acc2 = _mm512_fmadd_ps(_mm512_loadu_ps(&q[k + 32]), _mm512_loadu_ps(&k_row[k + 32]), acc2);
            acc3 = _mm512_fmadd_ps(_mm512_loadu_ps(&q[k + 48]), _mm512_loadu_ps(&k_row[k + 48]), acc3);
        }
        for (; k + 16 <= D; k += 16)
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(&q[k]), _mm512_loadu_ps(&k_row[k]), acc0);

        float dot = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(acc0, acc1), _mm512_add_ps(acc2, acc3)));
        scores[j] = dot * inv_sqrt;
    }

    // ── Phase 2: Branchless online softmax ───────────────────────────────
    // Step 2a: Find max (fully vectorized)
    __m512 v_max = _mm512_set1_ps(-1e30f);
    size_t j = 0;
    for (; j + 16 <= W; j += 16)
        v_max = _mm512_max_ps(v_max, _mm512_loadu_ps(&scores[j]));
    float max_s = _mm512_reduce_max_ps(v_max);
    for (; j < W; ++j)
        max_s = std::max(max_s, scores[j]);

    // Step 2b: exp(score - max) and accumulate sum
    __m512 v_maxb = _mm512_set1_ps(max_s);
    __m512 v_sum = _mm512_setzero_ps();
    j = 0;
    for (; j + 16 <= W; j += 16) {
        __m512 v_s = _mm512_sub_ps(_mm512_loadu_ps(&scores[j]), v_maxb);
        __m512 v_e = nca::simd::avx512::exp_ps(v_s);
        _mm512_storeu_ps(&scores[j], v_e);
        v_sum = _mm512_add_ps(v_sum, v_e);
    }
    float sum_exp = _mm512_reduce_add_ps(v_sum);
    for (; j < W; ++j) {
        scores[j] = std::exp(scores[j] - max_s);
        sum_exp += scores[j];
    }

    // Step 2c: Normalize — multiply by 1/sum (branchless rcp + Newton-Raphson)
    __m512 v_inv = _mm512_set1_ps(1.0f / sum_exp);
    j = 0;
    for (; j + 16 <= W; j += 16)
        _mm512_storeu_ps(&scores[j], _mm512_mul_ps(_mm512_loadu_ps(&scores[j]), v_inv));
    for (; j < W; ++j)
        scores[j] /= sum_exp;

    // ── Phase 3: Weighted V accumulation ─────────────────────────────────
    // out[k] = sum_j( scores[j] * v_cache[j, k] )
    // Zero the output accumulators (d_head/16 registers)
    constexpr size_t MAX_ACCUM = 128 / 16;  // 8 accumulators for d_head=128
    __m512 acc[MAX_ACCUM];
    for (size_t a = 0; a < D / 16; ++a)
        acc[a] = _mm512_setzero_ps();

    for (j = 0; j < W; ++j) [[likely]] {
        __m512 v_s = _mm512_set1_ps(scores[j]);
        const float* v_row = &v_cache[j * D];

        if constexpr (PF > 0) {
            if (j + PF < W)
                _mm_prefetch(reinterpret_cast<const char*>(&v_cache[(j + PF) * D]), _MM_HINT_T0);
        }

        for (size_t a = 0; a < D / 16; ++a)
            acc[a] = _mm512_fmadd_ps(v_s, _mm512_loadu_ps(&v_row[a * 16]), acc[a]);
    }

    // Store output
    for (size_t a = 0; a < D / 16; ++a)
        _mm512_storeu_ps(&out[a * 16], acc[a]);
}
#endif

void sla_step(
    const float* __restrict q,
    const float* __restrict k_cache,
    const float* __restrict v_cache,
    float* __restrict out,
    float* __restrict scores,
    SLAConfig cfg
) {
    if (simd::best_backend() == simd::Backend::AVX512) [[likely]] {
#if defined(__AVX512F__) || defined(_MSC_VER)
        sla_step_avx512(q, k_cache, v_cache, out, scores, cfg);
        return;
#endif
    }
    sla_step_scalar(q, k_cache, v_cache, out, scores, cfg);
}

} // namespace nca::layers
