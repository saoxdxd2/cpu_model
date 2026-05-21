#pragma once
// ============================================================================
// NCA — Dispatch Helpers
// core/simd/dispatch.hpp
// ============================================================================

#include <string>
#include <sstream>
#include <atomic>
#include <utility>

namespace nca::simd {

enum class Backend {
    Scalar,
    AVX2,
    AVX512
};

struct CPUInfo {
    bool has_avx2;
    bool has_avx512;
    bool has_vnni;
};

CPUInfo detect();
Backend best_backend();
void set_override_backend(Backend b);
void clear_override_backend();
const char* backend_name(Backend b) noexcept;

// ── OPTIMIZED DYNAMIC DISPATCHER ─────────────────────────────────────────────
template <typename FuncPtr>
class Dispatcher {
public:
    Dispatcher(FuncPtr s, FuncPtr a2, FuncPtr a512)
        : scalar(s), avx2(a2), avx512(a512), bound(nullptr) {}

    template <typename... Args>
    auto operator()(Args&&... args) const {
        auto* f = bound.load(std::memory_order_relaxed);
        if (f) [[likely]] return f(std::forward<Args>(args)...);
        return select()(std::forward<Args>(args)...);
    }

private:
    FuncPtr select() const {
        auto b = best_backend();
        FuncPtr f = scalar;
        if (b == Backend::AVX512) f = avx512;
        else if (b == Backend::AVX2) f = avx2;
        bound.store(f, std::memory_order_relaxed);
        return f;
    }

    FuncPtr scalar, avx2, avx512;
    mutable std::atomic<FuncPtr> bound;
};

#define NCA_DISPATCH_KERNEL(f512, f256, f_scalar, ...) do { \
    static const nca::simd::Dispatcher<decltype(&f_scalar)> _d(f_scalar, f256, f512); \
    return _d(__VA_ARGS__); \
} while (0)

} // namespace nca::simd
