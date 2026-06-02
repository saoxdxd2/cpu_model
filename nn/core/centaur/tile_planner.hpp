#pragma once
// ============================================================================
// CENTAUR — Cache-Aware Tile Planner
// centaur/tile_planner.hpp
//
// Step 2: Hardware-Driven Tile Geometry
//
// The key inversion: "hardware decides tile geometry, not the model."
//
// Given a CPUProfile, computes:
//   tile_size     = f(L1, L2, L3, bandwidth)
//   expert_blocks = f(SIMD_width, L2_capacity)
//   prefetch_depth = f(L3_size, bandwidth)
//
// All execution parameters are DERIVED from silicon, not from model config.
// ============================================================================

#include "core/centaur/cpu_fingerprint.hpp"
#include <algorithm>
#include <cmath>

namespace nca::centaur {

// ── TILE SHAPE: The atomic compute unit that fits in L1 ─────────────────────
struct TileShape {
    size_t M;                  // Rows per tile
    size_t N;                  // Cols per tile (SIMD-aligned)
    size_t K;                  // Inner reduction dimension
    size_t total_floats;       // M * N working set
    size_t total_bytes;        // total_floats * sizeof(float)
};

// ── PREFETCH SCHEDULE ───────────────────────────────────────────────────────
struct PrefetchSchedule {
    size_t depth;              // How many tiles ahead to prefetch
    size_t stride_bytes;       // Prefetch stride (cache-line aligned)
    bool   use_nt_stores;      // Non-temporal stores for write-only
    bool   use_l2_prefetch;    // Prefetch to L2 for streaming patterns
};

// ── EXPERT BLOCK LAYOUT (MoE Cache Compilation) ─────────────────────────────
struct ExpertBlockLayout {
    size_t num_resident_experts;    // How many experts fit in L2 simultaneously
    size_t expert_block_floats;     // Floats per expert weight slab
    size_t expert_block_bytes;      // Bytes per expert weight slab
    size_t top_k;                   // Number of experts activated per token
    size_t routing_batch_size;      // Tokens batched for routing amortization
};

// ── GEMM MICROKERNEL GEOMETRY ───────────────────────────────────────────────
struct MicrokernelShape {
    size_t mr;                 // Register-tile rows (output reuse)
    size_t nr;                 // Register-tile cols (broadcast reuse)
    size_t kr;                 // Unroll depth along K
    size_t simd_width;         // Elements per SIMD register
};

// ── COMPLETE EXECUTION PLAN ─────────────────────────────────────────────────
struct ExecutionPlan {
    TileShape         tile;
    PrefetchSchedule  prefetch;
    ExpertBlockLayout experts;
    MicrokernelShape  ukernel;

    // Overall strategy classification
    enum class Strategy : uint8_t {
        L1_RESIDENT,       // Everything fits L1d — zero prefetch, maximum ILP
        L2_STREAMING,      // Working set in L2, prefetch to L1 ahead
        L3_TILED,          // Tile for L1, stream from L3
        DRAM_BANDWIDTH     // Memory-bound — NT stores, aggressive prefetch
    } strategy;

    const char* strategy_name() const {
        switch (strategy) {
            case Strategy::L1_RESIDENT:   return "L1_RESIDENT";
            case Strategy::L2_STREAMING:  return "L2_STREAMING";
            case Strategy::L3_TILED:      return "L3_TILED";
            case Strategy::DRAM_BANDWIDTH:return "DRAM_BANDWIDTH";
        }
        return "UNKNOWN";
    }
};

// ── TILE PLANNER ────────────────────────────────────────────────────────────
// Pure function: CPUProfile → ExecutionPlan
// Zero side effects. Deterministic. Called once at startup per kernel shape.

inline ExecutionPlan plan_execution(
    const CPUProfile& cpu,
    size_t d_model,
    size_t n_experts = 0,
    size_t top_k     = 0
) {
    ExecutionPlan ep{};
    const size_t W = cpu.simd.simd_width_f32(); // 4, 8, or 16

    // ════════════════════════════════════════════════════════════════════════
    // 1. MICROKERNEL GEOMETRY
    //    Register tile shape is dictated by physical register file.
    //    AVX-512: 32 ZMM regs → 6×16 output tile + A/B panels + accumulators
    //    AVX2:    16 YMM regs → 6×8  output tile + A/B panels + accumulators
    // ════════════════════════════════════════════════════════════════════════
    ep.ukernel.simd_width = W;
    if (cpu.simd.avx512f) {
        // 6×16 output tile uses 6 regs for C accumulation,
        // 1 reg for A broadcast, 1 for B load, leaves headroom
        ep.ukernel.mr = 6;
        ep.ukernel.nr = W;    // 16
        ep.ukernel.kr = 4;    // Unroll K by 4 for FMA pipeline depth
    } else if (cpu.simd.avx2) {
        ep.ukernel.mr = 6;
        ep.ukernel.nr = W;    // 8
        ep.ukernel.kr = 4;
    } else {
        ep.ukernel.mr = 4;
        ep.ukernel.nr = W;    // 4
        ep.ukernel.kr = 2;
    }

    // ════════════════════════════════════════════════════════════════════════
    // 2. L1 TILE GEOMETRY
    //    tile_size = f(L1, L2, L3, bandwidth)
    //
    //    For a GEMM tile of shape [M, N] with K-reduction:
    //    Working set = M*K + K*N + M*N (A-panel + B-panel + C-tile)
    //    We solve for M, N such that this fits in 75% of L1d.
    // ════════════════════════════════════════════════════════════════════════
    size_t l1_usable   = cpu.L1d.usable_bytes(0.75f);
    size_t l2_usable   = cpu.L2.usable_bytes(0.75f);
    
    // N is always SIMD-aligned
    ep.tile.N = W;
    
    // K = inner dimension = d_model for most kernels
    ep.tile.K = d_model;

    // Solve for M: (M * K + K * N + M * N) * 4 <= l1_usable
    // M * (K + N) * 4 + K * N * 4 <= l1_usable
    // M <= (l1_usable - K * N * 4) / ((K + N) * 4)
    size_t b_panel_bytes  = ep.tile.K * ep.tile.N * sizeof(float);
    size_t per_row_bytes  = (ep.tile.K + ep.tile.N) * sizeof(float);
    
    if (per_row_bytes > 0 && b_panel_bytes < l1_usable) {
        size_t max_m = (l1_usable - b_panel_bytes) / per_row_bytes;
        // Round down to microkernel multiple
        max_m = (max_m / ep.ukernel.mr) * ep.ukernel.mr;
        ep.tile.M = (std::max)(max_m, ep.ukernel.mr);
    } else {
        // K is massive — use minimum tile, stream
        ep.tile.M = ep.ukernel.mr;
    }

    ep.tile.total_floats = ep.tile.M * ep.tile.N;
    ep.tile.total_bytes  = ep.tile.total_floats * sizeof(float);

    // ════════════════════════════════════════════════════════════════════════
    // 3. STRATEGY CLASSIFICATION
    //    Based on total working set vs. cache hierarchy
    // ════════════════════════════════════════════════════════════════════════
    size_t total_working_set = d_model * d_model * sizeof(float); // Full weight matrix
    
    if (total_working_set <= cpu.L1d.size_bytes / 2) {
        ep.strategy = ExecutionPlan::Strategy::L1_RESIDENT;
    } else if (total_working_set <= cpu.L2.size_bytes) {
        ep.strategy = ExecutionPlan::Strategy::L2_STREAMING;
    } else if (total_working_set <= cpu.L3.size_bytes) {
        ep.strategy = ExecutionPlan::Strategy::L3_TILED;
    } else {
        ep.strategy = ExecutionPlan::Strategy::DRAM_BANDWIDTH;
    }

    // ════════════════════════════════════════════════════════════════════════
    // 4. PREFETCH SCHEDULE
    //    Depth = f(L3 size, bandwidth)
    //    Larger L3 → more room to speculatively prefetch tiles
    //    Higher bandwidth → deeper prefetch pipeline profitable
    // ════════════════════════════════════════════════════════════════════════
    switch (ep.strategy) {
        case ExecutionPlan::Strategy::L1_RESIDENT:
            ep.prefetch.depth          = 0;
            ep.prefetch.stride_bytes   = 0;
            ep.prefetch.use_nt_stores  = false;
            ep.prefetch.use_l2_prefetch = false;
            break;

        case ExecutionPlan::Strategy::L2_STREAMING:
            ep.prefetch.depth          = 4;
            ep.prefetch.stride_bytes   = cpu.L1d.line_size;
            ep.prefetch.use_nt_stores  = false;
            ep.prefetch.use_l2_prefetch = false;
            break;

        case ExecutionPlan::Strategy::L3_TILED:
            ep.prefetch.depth          = cpu.is_big_l3() ? 8 : 4;
            ep.prefetch.stride_bytes   = cpu.L1d.line_size * 2;
            ep.prefetch.use_nt_stores  = false;
            ep.prefetch.use_l2_prefetch = true;
            break;

        case ExecutionPlan::Strategy::DRAM_BANDWIDTH:
            // Bandwidth-limited: maximize prefetch pipeline, use NT stores
            // for write-only buffers to avoid read-for-ownership
            ep.prefetch.depth          = cpu.is_big_l3() ? 16 : 8;
            ep.prefetch.stride_bytes   = cpu.L1d.line_size * 4;
            ep.prefetch.use_nt_stores  = true;
            ep.prefetch.use_l2_prefetch = true;
            break;
    }

    // ════════════════════════════════════════════════════════════════════════
    // 5. EXPERT BLOCK LAYOUT (MoE Cache Compilation)
    //    "MoE routing compiled into static memory blocks"
    //
    //    Key insight: each expert is a weight slab of [d_model, d_expert].
    //    We compute how many of these slabs fit simultaneously in L2,
    //    which determines the maximum number of cache-resident experts.
    // ════════════════════════════════════════════════════════════════════════
    if (n_experts > 0) {
        // Expert FFN dimension (typically d_model / n_experts * expansion)
        // For DeepSeekMoE-style micro-experts: each expert is small
        size_t d_expert = d_model; // Weight slab per expert
        ep.experts.expert_block_floats = d_expert;
        ep.experts.expert_block_bytes  = d_expert * sizeof(float);
        
        // How many experts fit in L2 with 75% occupancy?
        size_t l2_for_experts = l2_usable;
        ep.experts.num_resident_experts = l2_for_experts / ep.experts.expert_block_bytes;
        ep.experts.num_resident_experts = (std::min)(ep.experts.num_resident_experts, n_experts);
        ep.experts.num_resident_experts = (std::max)(ep.experts.num_resident_experts, size_t(1));

        ep.experts.top_k = top_k > 0 ? top_k : 
                           (cpu.simd.avx512f ? 8 : 4);
        
        // Routing batch: amortize gate computation
        // Larger L3 → can batch more tokens before flushing
        ep.experts.routing_batch_size = 
            (std::min)(size_t(64), cpu.L3.size_bytes / (d_model * sizeof(float) * 2));
        ep.experts.routing_batch_size = (std::max)(ep.experts.routing_batch_size, size_t(1));
    }

    return ep;
}

// ── CONVENIENCE: Plan for current hardware ──────────────────────────────────
inline ExecutionPlan plan_for_this_cpu(size_t d_model, size_t n_experts = 0, size_t top_k = 0) {
    return plan_execution(fingerprint(), d_model, n_experts, top_k);
}

} // namespace nca::centaur
