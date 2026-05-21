#pragma once
#include <memory>
#include <cstdlib>
#include <type_traits>

namespace nca::simd {

template <typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const {
        if (ptr) {
            if constexpr (!std::is_array_v<T> && !std::is_trivially_destructible_v<T>) {
                ptr->~T();
            }
            _aligned_free((void*)ptr);
        }
    }
};

// Specialization for arrays
template <typename T>
struct AlignedDeleter<T[]> {
    void operator()(T* ptr) const {
        if (ptr) {
            _aligned_free((void*)ptr);
        }
    }
};

template <typename T>
using aligned_unique_ptr = std::unique_ptr<T, AlignedDeleter<T>>;

// make_aligned_unique for non-arrays
template <typename T, typename... Args>
typename std::enable_if_t<!std::is_array_v<T>, aligned_unique_ptr<T>>
make_aligned_unique(Args&&... args) {
    void* ptr = _aligned_malloc(sizeof(T), 64);
    if (!ptr) throw std::bad_alloc();
    return aligned_unique_ptr<T>(new (ptr) T(std::forward<Args>(args)...));
}

// make_aligned_unique for arrays
template <typename T>
typename std::enable_if_t<std::is_array_v<T>, aligned_unique_ptr<T>>
make_aligned_unique(size_t size) {
    using ElementType = std::remove_extent_t<T>;
    void* ptr = _aligned_malloc(size * sizeof(ElementType), 64);
    if (!ptr) throw std::bad_alloc();
    return aligned_unique_ptr<T>(static_cast<ElementType*>(ptr));
}

} // namespace nca::simd
