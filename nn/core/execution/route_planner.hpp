#pragma once
// ============================================================================
// NCA -- Sequential Route Planner (Phase 9/10)
// core/execution/route_planner.hpp
//
// Implements sequential indexing ("planned route") for sparse token execution.
// Instead of traversing dynamic trees or allocating intermediate buffers in RAM,
// this planner precomputes an optimal execution sequence. The engine then
// iterates through these flat C-arrays, processing tokens depth-first
// across layers to strictly keep activations pinned in the L1 cache.
// ============================================================================

#include <cstddef>
#include <cstdint>

namespace nca::execution {

struct RoutePlan {
    size_t* active_indices = nullptr;
    size_t num_active = 0;
    size_t max_capacity = 0;

    RoutePlan() = default;
    explicit RoutePlan(size_t capacity);
    ~RoutePlan();

    RoutePlan(const RoutePlan&) = delete;
    RoutePlan& operator=(const RoutePlan&) = delete;

    RoutePlan(RoutePlan&& o) noexcept;
    RoutePlan& operator=(RoutePlan&& o) noexcept;
};

// Generates a sequential route based on an importance metric (e.g., SVD or norm).
// Tokens with metric >= threshold are added to the route.
// Returns the number of active tokens.
size_t plan_route_threshold(
    const float* metrics,
    size_t total_tokens,
    float threshold,
    RoutePlan& out_plan
);

// Generates a sequential route selecting exactly the Top-K tokens.
size_t plan_route_topk(
    const float* metrics,
    size_t total_tokens,
    size_t k,
    RoutePlan& out_plan
);

} // namespace nca::execution
