#pragma once
// ============================================================================
// CENTAUR — Hardware-Adaptive Execution Engine
// centaur/cpu_fingerprint.hpp
//
// Step 1: Runtime CPU Fingerprinting
//
// Probes the physical cache hierarchy, SIMD capabilities, and memory
// bandwidth characteristics at process startup. The resulting CPUProfile
// is the ROOT INPUT to the entire execution compilation pipeline.
//
// Control inversion: hardware constraints → execution topology.
// The model does NOT decide memory usage. The silicon does.
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <array>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <intrin.h>
#include <Windows.h>
#else
#include <cpuid.h>
#include <unistd.h>
#endif

namespace nca::centaur {

// ── CACHE LEVEL DESCRIPTOR ──────────────────────────────────────────────────
struct CacheLevel {
    size_t   size_bytes    = 0;      // Total capacity
    size_t   line_size     = 64;     // Bytes per cache line
    size_t   associativity = 0;      // N-way set associative
    size_t   sets          = 0;      // Number of sets
    bool     inclusive     = false;   // Inclusive of lower levels?

    // Derived: latency model (cycles, approximate)
    // Ice Lake: L1=4, L2=12, L3=~40, DRAM=~200
    uint32_t latency_cycles = 0;

    constexpr size_t num_lines() const { return size_bytes / line_size; }
    constexpr size_t usable_bytes(float occupancy = 0.75f) const {
        return static_cast<size_t>(size_bytes * occupancy);
    }
};

// ── SIMD CAPABILITY VECTOR ──────────────────────────────────────────────────
struct SimdProfile {
    bool     avx2          = false;
    bool     avx512f       = false;   // Foundation
    bool     avx512bw      = false;   // Byte/Word
    bool     avx512vl      = false;   // Vector Length
    bool     avx512vnni    = false;   // Vector Neural Network Instructions
    bool     avx512bf16    = false;   // BFloat16
    bool     amx_tile      = false;   // Advanced Matrix Extensions (Sapphire Rapids+)
    bool     amx_int8      = false;
    bool     amx_bf16      = false;

    // Effective SIMD width in floats (AVX2=8, AVX-512=16)
    constexpr size_t simd_width_f32() const {
        if (avx512f) return 16;
        if (avx2)    return 8;
        return 4; // SSE fallback
    }

    // GEMM microkernel register budget (zmm0-zmm31 for AVX-512, ymm0-ymm15 for AVX2)
    constexpr size_t register_file_floats() const {
        if (avx512f) return 32 * 16;  // 32 ZMM × 16 floats
        if (avx2)    return 16 * 8;   // 16 YMM × 8 floats
        return 16 * 4;
    }

    // Best available int8 throughput path
    constexpr bool has_int8_accel() const { return avx512vnni || amx_int8; }
};

// ── COMPLETE CPU PROFILE ────────────────────────────────────────────────────
struct CPUProfile {
    // Identity
    char     vendor[13]    = {};      // "GenuineIntel" / "AuthenticAMD"
    char     brand[49]     = {};      // Full brand string
    uint8_t  family        = 0;
    uint8_t  model         = 0;
    uint8_t  stepping      = 0;
    uint8_t  logical_cores = 1;
    uint8_t  physical_cores= 1;

    // Cache hierarchy (indexed: 0=L1d, 1=L2, 2=L3)
    CacheLevel L1d;
    CacheLevel L2;
    CacheLevel L3;

    // SIMD
    SimdProfile simd;

    // Memory bandwidth estimate (GB/s, single-core)
    // Computed via a microbenchmark or heuristic
    float    mem_bandwidth_gbps = 20.0f;

    // ── Derived convenience ─────────────────────────────────────────────
    constexpr size_t total_fast_cache() const {
        return L1d.size_bytes + L2.size_bytes;
    }

    constexpr bool is_big_l3() const {
        return L3.size_bytes >= (32ULL * 1024 * 1024);
    }
};

// ── FINGERPRINT FUNCTION ────────────────────────────────────────────────────
// Called ONCE at process startup. Probes everything via CPUID + OS APIs.
// Returns a const reference to the singleton profile.

namespace detail {

struct CpuidRegs {
    uint32_t eax, ebx, ecx, edx;
};

inline CpuidRegs cpuid_query(uint32_t leaf, uint32_t subleaf = 0) {
    CpuidRegs r{};
#ifdef _WIN32
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<uint32_t>(regs[0]);
    r.ebx = static_cast<uint32_t>(regs[1]);
    r.ecx = static_cast<uint32_t>(regs[2]);
    r.edx = static_cast<uint32_t>(regs[3]);
#else
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

inline bool os_xsave_avx() {
    auto r = cpuid_query(1);
    if (!(r.ecx & (1u << 27))) return false;
#ifdef _WIN32
    uint64_t xcr0 = _xgetbv(0);
#else
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    uint64_t xcr0 = (static_cast<uint64_t>(hi) << 32) | lo;
#endif
    return (xcr0 & 0x6) == 0x6;
}

inline bool os_xsave_avx512() {
    if (!os_xsave_avx()) return false;
#ifdef _WIN32
    uint64_t xcr0 = _xgetbv(0);
#else
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    uint64_t xcr0 = (static_cast<uint64_t>(hi) << 32) | lo;
#endif
    return (xcr0 & 0xE0) == 0xE0; // opmask + ZMM_Hi256 + Hi16_ZMM
}

inline void probe_cache_level(CPUProfile& p, uint32_t subleaf, const CpuidRegs& r) {
    uint32_t cache_type = r.eax & 0x1F;
    if (cache_type == 0) return; // No more caches
    
    uint32_t level      = (r.eax >> 5) & 0x7;
    uint32_t line_sz    = (r.ebx & 0xFFF) + 1;
    uint32_t partitions = ((r.ebx >> 12) & 0x3FF) + 1;
    uint32_t ways       = ((r.ebx >> 22) & 0x3FF) + 1;
    uint32_t sets       = r.ecx + 1;
    bool     inclusive   = (r.edx >> 1) & 1;
    size_t   total      = static_cast<size_t>(ways) * partitions * line_sz * sets;

    CacheLevel cl;
    cl.size_bytes    = total;
    cl.line_size     = line_sz;
    cl.associativity = ways;
    cl.sets          = sets;
    cl.inclusive     = inclusive;

    // Data cache (type 1) or unified (type 3)
    if (level == 1 && (cache_type == 1 || cache_type == 3)) {
        cl.latency_cycles = 4;
        p.L1d = cl;
    } else if (level == 2) {
        cl.latency_cycles = 12;
        p.L2 = cl;
    } else if (level == 3) {
        cl.latency_cycles = 40;
        p.L3 = cl;
    }
}

inline void probe_topology(CPUProfile& p) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    p.logical_cores = static_cast<uint8_t>(si.dwNumberOfProcessors);
    // Physical cores: query via GetLogicalProcessorInformation
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (len > 0) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buf.data(), &len)) {
            uint8_t cores = 0;
            for (auto& info : buf)
                if (info.Relationship == RelationProcessorCore) ++cores;
            p.physical_cores = cores > 0 ? cores : p.logical_cores;
        }
    }
#else
    p.logical_cores = static_cast<uint8_t>(sysconf(_SC_NPROCESSORS_ONLN));
    p.physical_cores = p.logical_cores; // Simplified
#endif
}

inline CPUProfile build_profile() {
    CPUProfile p{};

    // ── Vendor string ──
    auto r0 = cpuid_query(0);
    std::memcpy(p.vendor + 0, &r0.ebx, 4);
    std::memcpy(p.vendor + 4, &r0.edx, 4);
    std::memcpy(p.vendor + 8, &r0.ecx, 4);
    p.vendor[12] = '\0';

    // ── Brand string (leaves 0x80000002–0x80000004) ──
    auto max_ext = cpuid_query(0x80000000).eax;
    if (max_ext >= 0x80000004) {
        for (uint32_t i = 0; i < 3; ++i) {
            auto rb = cpuid_query(0x80000002 + i);
            std::memcpy(p.brand + i * 16 + 0,  &rb.eax, 4);
            std::memcpy(p.brand + i * 16 + 4,  &rb.ebx, 4);
            std::memcpy(p.brand + i * 16 + 8,  &rb.ecx, 4);
            std::memcpy(p.brand + i * 16 + 12, &rb.edx, 4);
        }
        p.brand[48] = '\0';
    }

    // ── Family / Model / Stepping ──
    auto r1 = cpuid_query(1);
    p.stepping  = r1.eax & 0xF;
    p.model     = ((r1.eax >> 4) & 0xF) | (((r1.eax >> 16) & 0xF) << 4);
    p.family    = ((r1.eax >> 8) & 0xF) + ((r1.eax >> 20) & 0xFF);

    // ── Cache hierarchy (CPUID leaf 4) ──
    for (uint32_t sub = 0; sub < 16; ++sub) {
        auto rc = cpuid_query(4, sub);
        if ((rc.eax & 0x1F) == 0) break;
        probe_cache_level(p, sub, rc);
    }

    // ── Fallback: if CPUID leaf 4 returned nothing (AMD uses leaf 0x8000001D) ──
    if (p.L1d.size_bytes == 0) {
        for (uint32_t sub = 0; sub < 16; ++sub) {
            auto rc = cpuid_query(0x8000001D, sub);
            if ((rc.eax & 0x1F) == 0) break;
            probe_cache_level(p, sub, rc);
        }
    }

    // ── Ultimate fallback: conservative defaults ──
    if (p.L1d.size_bytes == 0) { p.L1d.size_bytes = 32 * 1024;  p.L1d.line_size = 64; p.L1d.latency_cycles = 4;  }
    if (p.L2.size_bytes  == 0) { p.L2.size_bytes  = 256 * 1024; p.L2.line_size  = 64; p.L2.latency_cycles  = 12; }
    if (p.L3.size_bytes  == 0) { p.L3.size_bytes  = 6 * 1024 * 1024; p.L3.line_size = 64; p.L3.latency_cycles = 40; }

    // ── SIMD feature detection ──
    auto leaf7 = cpuid_query(7, 0);
    if (os_xsave_avx()) {
        p.simd.avx2     = (leaf7.ebx >> 5)  & 1;
        if (os_xsave_avx512()) {
            p.simd.avx512f    = (leaf7.ebx >> 16) & 1;
            p.simd.avx512bw   = (leaf7.ebx >> 30) & 1;
            p.simd.avx512vl   = (leaf7.ebx >> 31) & 1;
            p.simd.avx512vnni = (leaf7.ecx >> 11) & 1;
            p.simd.avx512bf16 = (leaf7.eax >> 5)  & 1; // leaf 7, sub 1 actually
        }
    }

    // ── AMX detection (leaf 7, sub 0) ──
    p.simd.amx_tile = (leaf7.edx >> 24) & 1;
    p.simd.amx_int8 = (leaf7.edx >> 25) & 1;
    p.simd.amx_bf16 = (leaf7.edx >> 22) & 1;

    // ── Topology ──
    probe_topology(p);

    // ── Memory bandwidth heuristic ──
    // Ice Lake ~40 GB/s dual-channel DDR4-3200
    // Alder Lake ~50 GB/s DDR5-4800
    // Per-core effective ≈ total / logical_cores * 1.5 (prefetch boost)
    float total_bw = (p.family >= 0x06 && p.model >= 0x9A) ? 50.0f : 40.0f;
    p.mem_bandwidth_gbps = total_bw / p.logical_cores * 1.5f;

    return p;
}

} // namespace detail

// ── SINGLETON ACCESS ────────────────────────────────────────────────────────
inline const CPUProfile& fingerprint() {
    static const CPUProfile profile = detail::build_profile();
    return profile;
}

} // namespace nca::centaur
