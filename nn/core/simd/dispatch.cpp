// ============================================================================
// NCA — Nano-Core Architecture
// core/simd/dispatch.cpp
//
// CPUID-based runtime detection for SIMD features.
//
// References:
//   Intel SDM Vol.2 — CPUID instruction
//   Leaf 1, ECX/EDX:  AVX, FMA
//   Leaf 7, Sub-leaf 0, EBX: AVX2, AVX-512F/BW/VL
//   Leaf 7, Sub-leaf 0, ECX: AVX-512 VNNI, VBMI2, BITALG, VPOPCNTDQ
// ============================================================================

#include "core/simd/dispatch.hpp"

#include <array>
#include <sstream>

#ifdef _MSC_VER
#include <intrin.h>  // __cpuid, __cpuidex
#else
#include <cpuid.h>   // __get_cpuid, __get_cpuid_count
#endif

namespace nca::simd {

// ── CPUID helper ──────────────────────────────────────────────────────────
namespace {

struct CpuidResult {
    uint32_t eax, ebx, ecx, edx;
};

CpuidResult cpuid(uint32_t leaf, uint32_t subleaf = 0) {
    CpuidResult r{};
#ifdef _MSC_VER
    std::array<int, 4> regs{};
    __cpuidex(regs.data(), static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = static_cast<uint32_t>(regs[0]);
    r.ebx = static_cast<uint32_t>(regs[1]);
    r.ecx = static_cast<uint32_t>(regs[2]);
    r.edx = static_cast<uint32_t>(regs[3]);
#else
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

/// Check if OS has enabled XSAVE for AVX (XCR0 bits 1,2)
/// and AVX-512 (XCR0 bits 5,6,7).
bool os_supports_avx() {
    auto r = cpuid(1);
    // OSXSAVE (ECX bit 27) must be set
    if (!(r.ecx & (1u << 27))) return false;

#ifdef _MSC_VER
    // Read XCR0 via _xgetbv
    uint64_t xcr0 = _xgetbv(0);
#else
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    uint64_t xcr0 = (static_cast<uint64_t>(hi) << 32) | lo;
#endif
    // Bits 1,2 = SSE + AVX state
    return (xcr0 & 0x6) == 0x6;
}

bool os_supports_avx512() {
#ifdef _MSC_VER
    uint64_t xcr0 = _xgetbv(0);
#else
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    uint64_t xcr0 = (static_cast<uint64_t>(hi) << 32) | lo;
#endif
    // Bits 5,6,7 = opmask, ZMM_Hi256, Hi16_ZMM
    return (xcr0 & 0xE0) == 0xE0;
}

CPUFeatures detect_impl() {
    CPUFeatures f{};

    if (!os_supports_avx()) return f;

    // Leaf 1: AVX, FMA
    auto leaf1 = cpuid(1);
    bool has_avx = (leaf1.ecx >> 28) & 1u;
    bool has_fma = (leaf1.ecx >> 12) & 1u;

    if (!has_avx) return f;

    // Leaf 7, sub-leaf 0: AVX2 and AVX-512 extensions
    auto leaf7 = cpuid(7, 0);

    f.avx2 = (leaf7.ebx >> 5) & 1u;
    f.fma  = has_fma;

    // AVX-512 requires OS support for ZMM registers
    if (os_supports_avx512()) {
        f.avx512f    = (leaf7.ebx >> 16) & 1u;  // bit 16
        f.avx512bw   = (leaf7.ebx >> 30) & 1u;  // bit 30
        f.avx512vl   = (leaf7.ebx >> 31) & 1u;  // bit 31

        f.avx512vnni        = (leaf7.ecx >> 11) & 1u; // bit 11
        f.avx512_vbmi2      = (leaf7.ecx >>  6) & 1u; // bit 6
        f.avx512_bitalg     = (leaf7.ecx >> 12) & 1u; // bit 12
        f.avx512_vpopcntdq  = (leaf7.ecx >> 14) & 1u; // bit 14
    }

    return f;
}

} // anonymous namespace

// ── Public API ────────────────────────────────────────────────────────────

const CPUFeatures& detect() {
    // Thread-safe one-time initialization (C++11 §6.7)
    static const CPUFeatures features = detect_impl();
    return features;
}

Backend best_backend() {
    const auto& f = detect();
    if (f.has_vnni())  return Backend::AVX512;
    if (f.has_avx2())  return Backend::AVX2;
    return Backend::Scalar;
}

const char* backend_name(Backend b) noexcept {
    switch (b) {
        case Backend::AVX512: return "AVX-512 (VNNI)";
        case Backend::AVX2:   return "AVX2 + FMA";
        case Backend::Scalar: return "Scalar (no SIMD)";
    }
    return "Unknown";
}

std::string CPUFeatures::to_string() const {
    std::ostringstream os;
    os << "CPU Features:\n";
    os << "  AVX2           : " << (avx2 ? "YES" : "no") << "\n";
    os << "  FMA            : " << (fma ? "YES" : "no") << "\n";
    os << "  AVX-512F       : " << (avx512f ? "YES" : "no") << "\n";
    os << "  AVX-512BW      : " << (avx512bw ? "YES" : "no") << "\n";
    os << "  AVX-512VL      : " << (avx512vl ? "YES" : "no") << "\n";
    os << "  AVX-512 VNNI   : " << (avx512vnni ? "YES" : "no") << "\n";
    os << "  AVX-512 VBMI2  : " << (avx512_vbmi2 ? "YES" : "no") << "\n";
    os << "  AVX-512 BITALG : " << (avx512_bitalg ? "YES" : "no") << "\n";
    os << "  AVX-512 VPOPCNTDQ : " << (avx512_vpopcntdq ? "YES" : "no") << "\n";
    os << "  ── Compound ──\n";
    os << "  VNNI ready     : " << (has_vnni() ? "YES" : "no") << "\n";
    os << "  AVX2 ready     : " << (has_avx2() ? "YES" : "no") << "\n";
    return os.str();
}

} // namespace nca::simd
