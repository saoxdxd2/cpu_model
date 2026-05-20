#pragma once
// ============================================================================
// NCA — Nano-Core Architecture
// core/simd/dispatch.hpp
//
// Runtime CPU feature detection via CPUID.
// Every math kernel queries best_backend() to select its SIMD path.
//
// Design:
//   - detect() runs CPUID once, caches result in a static local
//   - best_backend() returns the highest available: AVX512 > AVX2 > Scalar
//   - Individual feature flags available for fine-grained decisions
//     (e.g., VNNI available ≠ full AVX-512 available)
//
// Ice Lake i5-1035G1 specifics:
//   - 1× AVX-512 execution unit (port 0 only)
//   - Supports: AVX-512F, BW, VL, VNNI, VBMI2, BITALG, VPOPCNTDQ
//   - AVX2 executes on 2 ports → higher throughput for 256-bit ops
//   - Strategy: AVX2 constant baseline, AVX-512 VNNI boost for INT8 matmul
// ============================================================================

#include <cstdint>
#include <string>

namespace nca::simd {

// ── CPU Feature Flags ─────────────────────────────────────────────────────
// Detected once at startup via CPUID. Immutable after detection.
struct CPUFeatures {
    // AVX2 (baseline SIMD)
    bool avx2 = false;
    bool fma  = false;     // Fused multiply-add (always with AVX2 on modern CPUs)

    // AVX-512 family
    bool avx512f   = false; // Foundation (512-bit registers)
    bool avx512bw  = false; // Byte/Word operations
    bool avx512vl  = false; // Vector Length extensions (256-bit AVX-512)
    bool avx512vnni = false; // VPDPBUSD: INT8 multiply-accumulate

    // Ice Lake bonus extensions
    bool avx512_vbmi2      = false; // Variable-length bit manipulation
    bool avx512_bitalg     = false; // Bit algorithms (VPOPCNTB/W)
    bool avx512_vpopcntdq  = false; // Population count for DW/QW

    // ── Compound queries ──────────────────────────────────────────────

    /// True if VNNI INT8 matmul is available (the key acceleration path)
    /// Requires F + BW + VL + VNNI together.
    [[nodiscard]] bool has_vnni() const noexcept {
        return avx512f && avx512bw && avx512vl && avx512vnni;
    }

    /// True if AVX2 + FMA is available (baseline for all SIMD kernels)
    [[nodiscard]] bool has_avx2() const noexcept {
        return avx2 && fma;
    }

    /// Human-readable summary for logging at startup
    [[nodiscard]] std::string to_string() const;
};

// ── SIMD Backend Selection ────────────────────────────────────────────────
enum class Backend : uint8_t {
    Scalar = 0,  // No SIMD — fallback only (should never happen on target HW)
    AVX2   = 1,  // 256-bit — constant baseline, 2-port execution
    AVX512 = 2,  // 512-bit — 1-port boost, VNNI INT8 acceleration
};

/// Detect CPU features. Called once, result is cached.
/// Thread-safe (C++11 static local initialization guarantee).
const CPUFeatures& detect();

/// Returns the highest available backend for general dispatch.
Backend best_backend();

/// String name of a backend (for logging)
const char* backend_name(Backend b) noexcept;

} // namespace nca::simd
