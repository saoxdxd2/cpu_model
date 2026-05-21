// ============================================================================
// NCA -- Sequential Route Planner
// core/execution/route_planner.cpp
// ============================================================================

#include "core/execution/route_planner.hpp"
#include <malloc.h>
#include <algorithm>
#include <numeric>
#include <vector>

namespace nca::execution {

RoutePlan::RoutePlan(size_t capacity) : max_capacity(capacity), num_active(0) {
    active_indices = (size_t*)_aligned_malloc(capacity * sizeof(size_t), 64);
}

RoutePlan::~RoutePlan() {
    if (active_indices) {
        _aligned_free(active_indices);
    }
}

RoutePlan::RoutePlan(RoutePlan&& o) noexcept 
    : active_indices(o.active_indices), num_active(o.num_active), max_capacity(o.max_capacity) {
    o.active_indices = nullptr;
    o.num_active = 0;
    o.max_capacity = 0;
}

RoutePlan& RoutePlan::operator=(RoutePlan&& o) noexcept {
    if (this != &o) {
        this->~RoutePlan();
        active_indices = o.active_indices;
        num_active = o.num_active;
        max_capacity = o.max_capacity;
        o.active_indices = nullptr;
        o.num_active = 0;
        o.max_capacity = 0;
    }
    return *this;
}

size_t plan_route_threshold(
    const float* metrics,
    size_t total_tokens,
    float threshold,
    RoutePlan& out_plan
) {
    size_t active_count = 0;
    for (size_t i = 0; i < total_tokens; ++i) {
        if (metrics[i] >= threshold && active_count < out_plan.max_capacity) {
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
    size_t actual_k = std::min(k, total_tokens);
    actual_k = std::min(actual_k, out_plan.max_capacity);

    // Using a temporary vector to sort indices by metric
    std::vector<size_t> indices(total_tokens);
    std::iota(indices.begin(), indices.end(), 0);

    // Partial sort to find top K
    std::partial_sort(
        indices.begin(), 
        indices.begin() + actual_k, 
        indices.end(),
        [&metrics](size_t a, size_t b) {
            return metrics[a] > metrics[b]; // Descending order
        }
    );

    // Write to the flat C-array to guarantee sequential, branchless indexing later
    for (size_t i = 0; i < actual_k; ++i) {
        out_plan.active_indices[i] = indices[i];
    }
    
    // Sort the active indices themselves so memory access is strictly forward/ascending
    // This maximizes hardware prefetching efficiency
    std::sort(out_plan.active_indices, out_plan.active_indices + actual_k);

    out_plan.num_active = actual_k;
    return actual_k;
}

} // namespace nca::execution
