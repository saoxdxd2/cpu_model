// ============================================================================
// NCA — Hardware Bottleneck & Cycle Profiler
// tests/test_profiling.cpp
//
// Identifies if functions are Compute-Bound or Memory-Bandwidth Bound
// using exact CPU cycles (__rdtsc) and nanosecond precision.
// ============================================================================

#include "core/normalization.hpp"
#include "core/activations.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <malloc.h> // For _aligned_malloc

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

template <typename Func>
void profile_function(const std::string& name, size_t d_size, Func f) {
    constexpr size_t WARMUP = 5000;
    constexpr size_t ITERS = 100000;
    
    // Warmup branch predictor and cache
    for(size_t i=0; i<WARMUP; i++) f();

    uint64_t start_cycles = __rdtsc();
    auto t0 = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < ITERS; ++i) {
        f();
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    uint64_t end_cycles = __rdtsc();
    
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ns_per_iter = (total_ms * 1e6) / ITERS;
    
    uint64_t total_cycles = end_cycles - start_cycles;
    double cycles_per_iter = static_cast<double>(total_cycles) / ITERS;
    double cycles_per_elem = cycles_per_iter / d_size;
    
    std::cout << "  [" << name << "]\n";
    std::cout << "    Latency       : " << ns_per_iter << " nanoseconds / call\n";
    std::cout << "    CPU Cycles    : " << cycles_per_iter << " cycles / call\n";
    std::cout << "    Cycles / Elem : " << cycles_per_elem << " cycles / element\n";
    
    if (cycles_per_elem < 0.5) {
        std::cout << "    Bottleneck    : MEMORY BOUND (Hitting DDR4/L2 Bandwidth limit)\n\n";
    } else {
        std::cout << "    Bottleneck    : COMPUTE BOUND (Hitting ALU/FPU limits)\n\n";
    }
}

int main() {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  NCA — Cycle-Accurate Hardware Profiler  ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";

    constexpr size_t D = 8192;
    
    // Ensure perfectly 64-byte aligned memory to trigger AVX-512 non-temporal streams
    float* in_ptr  = (float*)_aligned_malloc(D * sizeof(float), 64);
    float* w_ptr   = (float*)_aligned_malloc(D * sizeof(float), 64);
    float* out_ptr = (float*)_aligned_malloc(D * sizeof(float), 64);

    for(size_t i=0; i<D; i++) {
        in_ptr[i] = 1.0f;
        w_ptr[i] = 1.0f;
        out_ptr[i] = 0.0f;
    }

    std::span<const float> in_s(in_ptr, D);
    std::span<const float> w_s(w_ptr, D);
    std::span<float> out_s(out_ptr, D);

    std::cout << "[Profiling Vector Length D=" << D << " (" << (D * 4) / 1024 << " KB)]\n";
    
    profile_function("nca::math::rmsnorm (Non-Temporal)", D, [&]() {
        nca::math::rmsnorm(out_s, in_s, w_s);
    });

    profile_function("nca::math::silu (Non-Temporal)", D, [&]() {
        nca::math::silu(out_s);
    });

    _aligned_free(in_ptr);
    _aligned_free(w_ptr);
    _aligned_free(out_ptr);

    return 0;
}
