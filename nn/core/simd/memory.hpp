#pragma once
// ============================================================================
// NCA — Aligned Memory Management
// core/simd/memory.hpp
// ============================================================================

#include <memory>
#include <stdexcept>

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

namespace nca::simd {

struct AlignedDeleter {
    void operator()(void* p) const noexcept {
        if (!p) return;
#ifdef _WIN32
        _aligned_free(p);
#else
        free(p);
#endif
    }
};

template <typename T>
using aligned_unique_ptr = std::unique_ptr<T, AlignedDeleter>;

template <typename T>
inline auto make_aligned_unique(size_t count, size_t alignment = 64) {
    if (count == 0) return aligned_unique_ptr<T[]>(nullptr);
#ifdef _WIN32
    void* p = _aligned_malloc(count * sizeof(T), alignment);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, count * sizeof(T)) != 0) p = nullptr;
#endif
    if (!p) throw std::bad_alloc();
    return aligned_unique_ptr<T[]>(static_cast<T*>(p));
}

} // namespace nca::simd
