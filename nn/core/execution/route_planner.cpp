// ============================================================================
// NCA -- Sequential Route Planner
// core/execution/route_planner.cpp
// ============================================================================

#include "core/execution/route_planner.hpp"
#include <algorithm>
#include <numeric>
#include <vector>

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
        // We use AVX-512 to copy entire tokens into the contiguous buffer.
        size_t d = 0;
#if defined(__AVX512F__) || defined(_MSC_VER)
        for (; d + 15 < d_model; d += 16) {
            _mm512_storeu_ps(&p_dst[d], _mm512_loadu_ps(&p_src[d]));
        }
#endif
        // Scalar tail (unrolled 4x)
        for (; d + 3 < d_model; d += 4) {
            p_dst[d] = p_src[d];
            p_dst[d+1] = p_src[d+1];
            p_dst[d+2] = p_src[d+2];
            p_dst[d+3] = p_src[d+3];
        }
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

#if defined(__AVX512F__) || defined(_MSC_VER)
    __m512 v_thresh = _mm512_set1_ps(threshold);
    
    size_t i = 0;
    for (; i + 15 < total_tokens; i += 16) [[likely]] {
        __m512 v_m = _mm512_loadu_ps(&metrics[i]);
        __mmask16 mask = _mm512_cmp_ps_mask(v_m, v_thresh, _CMP_GE_OQ);
        
        if (mask == 0) continue;

        int cnt = _mm_popcnt_u32(static_cast<uint32_t>(mask));
        if (active_count + cnt > capacity) break;

        __m512i v_idx = _mm512_setr_epi32(
            (int)i,   (int)i+1, (int)i+2, (int)i+3,
            (int)i+4, (int)i+5, (int)i+6, (int)i+7,
            (int)i+8, (int)i+9, (int)i+10,(int)i+11,
            (int)i+12,(int)i+13,(int)i+14,(int)i+15
        );

        _mm512_mask_compressstoreu_epi32(&out_plan.active_indices[active_count], mask, v_idx);
        active_count += cnt;
    }
#else
    size_t i = 0;
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
    // Instead of sorting (O(N log N)) or partial sorting (O(N log K)), 
    // we use Quick-Select (O(N)) to find the K-th largest element.
    // This is the "Google/V8" approach to finding hot items in a stream.
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

    // After nth_element, indices[0...actual_k-1] are the top-K in arbitrary order.
    // We physically copy them to the plan.
    for (size_t i = 0; i < actual_k; ++i) {
        out_plan.active_indices[i] = indices[i];
    }
    
    // Sort output indices to ensure sequential memory access in downstream cores.
    std::sort(out_plan.active_indices, out_plan.active_indices + actual_k);

    out_plan.num_active = actual_k;
    return actual_k;
}

} // namespace nca::execution
