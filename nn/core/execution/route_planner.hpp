#pragma once
// ============================================================================
// NCA -- Sequential Route Planner
// core/execution/route_planner.hpp
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <memory>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace nca::execution {

struct AlignedDeleter {
    void operator()(void* p) const noexcept {
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

template <typename T>
using aligned_unique_ptr = std::unique_ptr<T, AlignedDeleter>;

struct RoutePlan {
    size_t* active_indices = nullptr;
    size_t num_active = 0;
    size_t max_capacity = 0;

    // Ownership handle
    aligned_unique_ptr<size_t[]> active_indices_ptr;

    RoutePlan() = default;
    explicit RoutePlan(size_t capacity);
    ~RoutePlan() = default;

    RoutePlan(const RoutePlan&) = delete;
    RoutePlan& operator=(const RoutePlan&) = delete;

    RoutePlan(RoutePlan&& o) noexcept = default;
    RoutePlan& operator=(RoutePlan&& o) noexcept = default;
};

size_t plan_route_threshold(
    const float* __restrict metrics,
    size_t total_tokens,
    float threshold,
    RoutePlan& out_plan
);

size_t plan_route_topk(
    const float* metrics,
    size_t total_tokens,
    size_t k,
    RoutePlan& out_plan
);

} // namespace nca::execution
