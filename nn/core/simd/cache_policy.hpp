#pragma once
// ============================================================================
// NCA -- Compile-Time Cache Policy Engine
// core/simd/cache_policy.hpp
//
// The Triton/XLA trick for CPU: compute the optimal tiling strategy,
// prefetch distance, and store policy at COMPILE TIME via constexpr.
// Zero branches at runtime. The compiler emits ONLY the optimal path.
//
// Usage:
//   constexpr auto policy = nca::simd::cache_policy<WorkingSetBytes>();
//   policy.tile_size   -- how many elements per L1 tile
//   policy.prefetch_dist -- how far ahead to prefetch (in cache lines)
//   policy.Strategy    -- L1_HOT / L2_STREAM / DDR4_NT
// ============================================================================

#include <cstddef>

namespace nca::simd {

// Ice Lake i5-1035G1 cache hierarchy (per-core, exact)
inline constexpr size_t L1_SIZE   = 32 * 1024;   // 32 KB L1d
inline constexpr size_t L2_SIZE   = 256 * 1024;   // 256 KB L2
inline constexpr size_t L3_SIZE   = 6 * 1024 * 1024; // 6 MB L3 shared
inline constexpr size_t CACHE_LINE = 64;

enum class CacheStrategy : int {
    L1_HOT,       // Working set fits L1 → no prefetch, tight loop
    L2_STREAM,    // Working set fits L2 → prefetch to L1 ahead, regular stores
    DDR4_NT       // Working set exceeds L2 → tile for L1, NT stores for write-only
};

// Compile-time policy: given a working set in bytes, compute the optimal strategy.
template <size_t WorkingSetBytes>
struct CachePolicy {
    static constexpr CacheStrategy strategy =
        (WorkingSetBytes <= L1_SIZE / 2) ? CacheStrategy::L1_HOT :
        (WorkingSetBytes <= L2_SIZE)     ? CacheStrategy::L2_STREAM :
                                          CacheStrategy::DDR4_NT;

    // Tile size: how many CACHE LINES fit in half of L1 (leave room for registers)
    // For L1_HOT: no tiling needed (everything fits)
    // For L2_STREAM: tile to fill L1 with prefetched data
    // For DDR4_NT: tile aggressively to keep hot data in L1
    static constexpr size_t tile_lines =
        (strategy == CacheStrategy::L1_HOT) ? WorkingSetBytes / CACHE_LINE :
        L1_SIZE / (2 * CACHE_LINE);

    // Prefetch distance in cache lines
    static constexpr size_t prefetch_dist =
        (strategy == CacheStrategy::L1_HOT)   ? 0 :
        (strategy == CacheStrategy::L2_STREAM) ? 4 :
                                                  8;

    // Whether to use non-temporal stores for write-only buffers
    static constexpr bool use_nt_stores = (strategy == CacheStrategy::DDR4_NT);

    // Human-readable name for diagnostics
    static constexpr const char* name() {
        if constexpr (strategy == CacheStrategy::L1_HOT)   return "L1_HOT";
        if constexpr (strategy == CacheStrategy::L2_STREAM) return "L2_STREAM";
        return "DDR4_NT";
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
