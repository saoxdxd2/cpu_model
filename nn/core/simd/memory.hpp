#pragma once
#include <memory>
#include <cstdlib>
#include <type_traits>
#include <new>

namespace nca::simd {

template <typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const {
        if (ptr) {
            if constexpr (!std::is_array_v<T>) {
                if constexpr (!std::is_trivially_destructible_v<T>) {
                    ptr->~T();
                }
            }
            _aligned_free((void*)ptr);
        }
    }
};

template <typename T>
struct AlignedDeleter<T[]> {
    size_t count = 0;
    AlignedDeleter() = default;
    explicit AlignedDeleter(size_t n) : count(n) {}

    void operator()(T* ptr) const {
        if (ptr) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                for (size_t i = 0; i < count; ++i) {
                    ptr[i].~T();
                }
            }
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

// make_aligned_unique for arrays (Now correctly calls constructors)
template <typename T>
typename std::enable_if_t<std::is_array_v<T>, aligned_unique_ptr<T>>
make_aligned_unique(size_t count) {
    using ElementType = std::remove_extent_t<T>;
    void* ptr = _aligned_malloc(count * sizeof(ElementType), 64);
    if (!ptr) throw std::bad_alloc();
    
    ElementType* e_ptr = static_cast<ElementType*>(ptr);
    size_t i = 0;
    try {
        for (; i < count; ++i) {
            new (&e_ptr[i]) ElementType();
        }
    } catch (...) {
        for (size_t j = 0; j < i; ++j) e_ptr[j].~ElementType();
        _aligned_free(ptr);
        throw;
    }
    
    return aligned_unique_ptr<T>(e_ptr, AlignedDeleter<T>(count));
}

} // namespace nca::simd
