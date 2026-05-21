// ============================================================================
// NCA — Nano-Core Architecture
// core/simd/dispatch.cpp
// ============================================================================

#include "core/simd/dispatch.hpp"
#include <array>
#include <sstream>
#include <optional>

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

namespace nca::simd {

namespace {

struct CpuidResult {
    uint32_t eax, ebx, ecx, edx;
};

CpuidResult cpuid(uint32_t leaf, uint32_t subleaf = 0) {
    CpuidResult r{};
#ifdef _WIN32
    int regs[4];
    __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
    r.eax = regs[0]; r.ebx = regs[1]; r.ecx = regs[2]; r.edx = regs[3];
#else
    __cpuid_count(leaf, subleaf, r.eax, r.ebx, r.ecx, r.edx);
#endif
    return r;
}

bool os_supports_avx() {
    auto r = cpuid(1);
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

bool os_supports_avx512() {
    if (!os_supports_avx()) return false;
#ifdef _WIN32
    uint64_t xcr0 = _xgetbv(0);
#else
    uint32_t lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    uint64_t xcr0 = (static_cast<uint64_t>(hi) << 32) | lo;
#endif
    return (xcr0 & 0xE0) == 0xE0;
}

CPUInfo detect_impl() {
    CPUInfo f{false, false, false};
    if (!os_supports_avx()) return f;

    auto leaf7 = cpuid(7, 0);
    f.has_avx2 = (leaf7.ebx >> 5) & 1u;

    if (os_supports_avx512()) {
        f.has_avx512 = (leaf7.ebx >> 16) & 1u;
        f.has_vnni   = (leaf7.ecx >> 11) & 1u;
    }
    return f;
}

} // namespace

CPUInfo detect() {
    static const CPUInfo info = detect_impl();
    return info;
}

static std::optional<Backend> g_override_backend;

void set_override_backend(Backend b) { g_override_backend = b; }
void clear_override_backend() { g_override_backend.reset(); }

Backend best_backend() {
    if (g_override_backend) return *g_override_backend;
    auto f = detect();
    if (f.has_vnni && f.has_avx512) return Backend::AVX512;
    if (f.has_avx2) return Backend::AVX2;
    return Backend::Scalar;
}

const char* backend_name(Backend b) noexcept {
    switch (b) {
        case Backend::AVX512: return "AVX-512 (VNNI)";
        case Backend::AVX2:   return "AVX2";
        default:              return "Scalar";
    }
}

} // namespace nca::simd
