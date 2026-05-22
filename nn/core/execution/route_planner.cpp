// ============================================================================
// NCA -- Sequential Route Planner
// core/execution/route_planner.cpp
// ============================================================================

#include "core/execution/route_planner.hpp"
#include <algorithm>
#include <numeric>
#include <vector>
#include <immintrin.h>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace nca::execution {

RoutePlan::RoutePlan(size_t capacity) : max_capacity(capacity), num_active(0) {
    active_indices_ptr.reset((size_t*)_aligned_malloc(capacity * sizeof(size_t), 64));
    active_indices = active_indices_ptr.get();
}

void shuffle_active_tokens(
    const float* __restrict src,
    float* __restrict dst,
    const RoutePlan& plan,
    size_t d_model
) {
    const auto count = plan.num_active;
    const auto* indices = plan.active_indices;

    for (size_t i = 0; i < count; ++i) [[likely]] {
        const auto idx = indices[i];
        const float* p_src = src + idx * d_model;
        float* p_dst = dst + i * d_model;

        // ── SIMD-Accelerated Contiguous Copy ─────────────────────────────────
        size_t d = 0;
#if defined(__AVX512F__)
        for (; d + 15 < d_model; d += 16) {
            _mm512_storeu_ps(&p_dst[d], _mm512_loadu_ps(&p_src[d]));
        }
#else
        for (; d + 7 < d_model; d += 8) {
            _mm256_storeu_ps(&p_dst[d], _mm256_loadu_ps(&p_src[d]));
        }
#endif
        for (; d < d_model; ++d) p_dst[d] = p_src[d];
    }
}

size_t plan_route_threshold(
    const float* __restrict metrics,
    size_t total_tokens,
    float threshold,
    RoutePlan& out_plan
) {
    size_t active_count = 0;
    const auto capacity = out_plan.max_capacity;

    size_t i = 0;
#if defined(__AVX512F__)
    __m512 v_thresh = _mm512_set1_ps(threshold);
    
    for (; i + 15 < total_tokens; i += 16) [[likely]] {
        __m512 v_m = _mm512_loadu_ps(&metrics[i]);
        __mmask16 mask = _mm512_cmp_ps_mask(v_m, v_thresh, _CMP_GE_OQ);
        
        if (mask == 0) continue;

        uint32_t m = static_cast<uint32_t>(mask);
        while (m != 0 && active_count < capacity) {
#ifdef _MSC_VER
            unsigned long bit_idx;
            _BitScanForward(&bit_idx, m);
            out_plan.active_indices[active_count++] = i + bit_idx;
            m &= ~(1U << bit_idx);
#else
            int bit_idx = __builtin_ctz(m);
            out_plan.active_indices[active_count++] = i + bit_idx;
            m &= m - 1;
#endif
        }
        if (active_count >= capacity) break;
    }
#else
    __m256 v_thresh = _mm256_set1_ps(threshold);
    
    for (; i + 7 < total_tokens; i += 8) [[likely]] {
        __m256 v_m = _mm256_loadu_ps(&metrics[i]);
        __m256 cmp = _mm256_cmp_ps(v_m, v_thresh, _CMP_GE_OQ);
        uint32_t mask = (uint32_t)_mm256_movemask_ps(cmp);
        
        if (mask == 0) continue;

        while (mask != 0 && active_count < capacity) {
#ifdef _MSC_VER
            unsigned long bit_idx;
            _BitScanForward(&bit_idx, mask);
            out_plan.active_indices[active_count++] = i + bit_idx;
            mask &= ~(1U << bit_idx);
#else
            int bit_idx = __builtin_ctz(mask);
            out_plan.active_indices[active_count++] = i + bit_idx;
            mask &= mask - 1;
#endif
        }
        if (active_count >= capacity) break;
    }
#endif

    for (; i < total_tokens; ++i) {
        if (metrics[i] >= threshold && active_count < capacity) {
            out_plan.active_indices[active_count++] = i;
        }
    }
    
    out_plan.num_active = active_count;
    return active_count;
}

size_t plan_route_topk(
    const float* metrics,
    size_t total_tokens,
    size_t k,
    RoutePlan& out_plan
) {
    auto actual_k = std::min({k, total_tokens, out_plan.max_capacity});
    if (actual_k == 0) return 0;

    // ── COMPLEXITY DESTRUCTION: O(N) QUICK-SELECT ───────────────────────────
    std::vector<size_t> indices(total_tokens);
    std::iota(indices.begin(), indices.end(), 0);

    std::nth_element(
        indices.begin(), 
        indices.begin() + actual_k, 
        indices.end(),
        [&metrics](auto a, auto b) {
            return metrics[a] > metrics[b];
        }
    );

    for (size_t i = 0; i < actual_k; ++i) {
        out_plan.active_indices[i] = indices[i];
    }
    
    std::sort(out_plan.active_indices, out_plan.active_indices + actual_k);

    out_plan.num_active = actual_k;
    return actual_k;
}

} // namespace nca::execution
