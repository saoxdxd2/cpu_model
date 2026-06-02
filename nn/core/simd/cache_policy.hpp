#pragma once
// ============================================================================
// NCA -- Cache Policy Engine
// core/simd/cache_policy.hpp
//
// Two modes:
//   1. COMPILE-TIME (original): CachePolicy<WorkingSetBytes>
//      Uses conservative defaults. Zero branches at runtime.
//      Best when working set is known at compile time.
//
//   2. RUNTIME (CENTAUR): RuntimeCachePolicy
//      Uses actual cache sizes probed via CPUID.
//      Best when working set depends on model config.
//
// Both coexist — existing kernels using CachePolicy<> are unchanged.
// ============================================================================

#include <cstddef>

namespace nca::simd {

// ── Conservative compile-time defaults ──────────────────────────────────────
// These are lower bounds that work on ANY modern x86.
// CENTAUR runtime queries override these with actual values.
inline constexpr size_t L1_SIZE   = 32 * 1024;     // Conservative L1d
inline constexpr size_t L2_SIZE   = 256 * 1024;    // Conservative L2
inline constexpr size_t L3_SIZE   = 6 * 1024 * 1024; // Conservative L3
inline constexpr size_t CACHE_LINE = 64;

enum class CacheStrategy : int {
    L1_HOT,       // Working set fits L1 → no prefetch, tight loop
    L2_STREAM,    // Working set fits L2 → prefetch to L1 ahead, regular stores
    DDR4_NT       // Working set exceeds L2 → tile for L1, NT stores for write-only
};

// ── COMPILE-TIME POLICY (original, backward-compatible) ─────────────────────
template <size_t WorkingSetBytes>
struct CachePolicy {
    static constexpr CacheStrategy strategy =
        (WorkingSetBytes <= L1_SIZE / 2) ? CacheStrategy::L1_HOT :
        (WorkingSetBytes <= L2_SIZE)     ? CacheStrategy::L2_STREAM :
                                          CacheStrategy::DDR4_NT;

    static constexpr size_t tile_lines =
        (strategy == CacheStrategy::L1_HOT) ? WorkingSetBytes / CACHE_LINE :
        L1_SIZE / (2 * CACHE_LINE);

    static constexpr size_t prefetch_dist =
        (strategy == CacheStrategy::L1_HOT)   ? 0 :
        (strategy == CacheStrategy::L2_STREAM) ? 4 :
                                                  8;

    static constexpr bool use_nt_stores = (strategy == CacheStrategy::DDR4_NT);

    static constexpr const char* name() {
        if constexpr (strategy == CacheStrategy::L1_HOT)   return "L1_HOT";
        if constexpr (strategy == CacheStrategy::L2_STREAM) return "L2_STREAM";
        return "DDR4_NT";
    }
};

// ── RUNTIME POLICY (CENTAUR-backed) ─────────────────────────────────────────
// Queries actual cache sizes at runtime. Use when working set is dynamic.
struct RuntimeCachePolicy {
    CacheStrategy strategy;
    size_t tile_lines;
    size_t prefetch_dist;
    bool   use_nt_stores;

    // Construct from actual cache sizes + working set
    static RuntimeCachePolicy compute(
        size_t working_set_bytes,
        size_t l1_size,
        size_t l2_size,
        size_t cache_line = 64
    ) {
        RuntimeCachePolicy p{};
        if (working_set_bytes <= l1_size / 2) {
            p.strategy      = CacheStrategy::L1_HOT;
            p.tile_lines    = working_set_bytes / cache_line;
            p.prefetch_dist = 0;
            p.use_nt_stores = false;
        } else if (working_set_bytes <= l2_size) {
            p.strategy      = CacheStrategy::L2_STREAM;
            p.tile_lines    = l1_size / (2 * cache_line);
            p.prefetch_dist = 4;
            p.use_nt_stores = false;
        } else {
            p.strategy      = CacheStrategy::DDR4_NT;
            p.tile_lines    = l1_size / (2 * cache_line);
            p.prefetch_dist = 8;
            p.use_nt_stores = true;
        }
        return p;
    }

    const char* name() const {
        switch (strategy) {
            case CacheStrategy::L1_HOT:   return "L1_HOT";
            case CacheStrategy::L2_STREAM: return "L2_STREAM";
            case CacheStrategy::DDR4_NT:  return "DDR4_NT";
        }
        return "UNKNOWN";
    }
};

// ── Branchless tail masking ──────────────────────────────────────────────────
// Instead of `if (remaining < 16)`, compute a mask that handles partial vectors.
// This is the branchless decision tree the user asked for.
inline __mmask16 tail_mask(size_t remaining) {
    // _bzhi_u32(0xFFFF, n) sets the bottom n bits. Zero branches.
    return static_cast<__mmask16>((remaining >= 16) ? 0xFFFF : (1u << remaining) - 1u);
}

// ── Compile-time working set calculator ──────────────────────────────────────
// Compute the total memory footprint of a kernel given its array count and size.
template <size_t NumArrays, size_t ElementsPerArray, size_t BytesPerElement = 4>
inline constexpr size_t working_set_bytes = NumArrays * ElementsPerArray * BytesPerElement;

// Convenience aliases for common kernel shapes
template <size_t D>
using GLRPolicy = CachePolicy<working_set_bytes<4, D>>;  // h, alpha, beta, x

template <size_t D, size_t S = 16>
using SSMPolicy = CachePolicy<working_set_bytes<2, D * S> + working_set_bytes<3, D>>;  // h+A big, x+y+B+C small

} // namespace nca::simd
