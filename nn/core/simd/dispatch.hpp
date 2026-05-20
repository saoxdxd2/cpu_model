#pragma once
// ============================================================================
// NCA — Dispatch Helpers
// core/simd/dispatch.hpp
// ============================================================================

#include <string>

namespace nca::simd {

struct CPUFeatures {
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512vl = false;
    bool avx512vnni = false;
    bool avx512_vbmi2 = false;
    bool avx512_bitalg = false;
    bool avx512_vpopcntdq = false;

    bool has_avx2() const { return avx2 && fma; }
    bool has_vnni() const { return avx512f && avx512bw && avx512vl && avx512vnni; }
    
    std::string to_string() const;
};

const CPUFeatures& detect();

enum class Backend {
    Scalar = 0,
    AVX2 = 1,
    AVX512 = 2
};

Backend best_backend();
const char* backend_name(Backend b) noexcept;

#define NCA_DISPATCH_KERNEL(f512, f256, f_scalar, ...) \
    do { \
        nca::simd::Backend _b_ = nca::simd::best_backend(); \
        if (_b_ == nca::simd::Backend::AVX512) [[likely]] { return f512(__VA_ARGS__); } \
        if (_b_ == nca::simd::Backend::AVX2) { return f256(__VA_ARGS__); } \
        return f_scalar(__VA_ARGS__); \
    } while (0)

} // namespace nca::simd
