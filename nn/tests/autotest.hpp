#pragma once
// ============================================================================
// NCA -- C++ Template Metaprogramming Auto-Benchmark Engine
// ============================================================================

#include "core/simd/dispatch.hpp"
#include "core/simd/memory.hpp"
#include "core/linalg/mx_linear.hpp"
#include "core/layers/ssm.hpp"
#include "core/layers/sla.hpp"
#include "core/layers/halting.hpp"
#include "core/vision/scanner.hpp"
#include <iostream>
#include <tuple>
#include <utility>
#include <span>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace nca::testing {

constexpr size_t D = 8192;

template <typename T> struct StripRestrict              { using type = T; };
template <typename T> struct StripRestrict<T* __restrict> { using type = T*; };
template <typename T> using strip_t = typename StripRestrict<T>::type;

inline auto hash_bytes(const void* data, size_t len) {
    if (!data) return 0ULL;
    uint64_t h = 14695981039346656037ULL;
    const auto* ptr = (const uint8_t*)data;
    for (size_t i = 0; i < len; ++i) {
        h ^= ptr[i];
        h *= 1099511628211ULL;
    }
    return h;
}

template <typename T> struct ArgGen;

template <> struct ArgGen<float*> {
    nca::simd::aligned_unique_ptr<float[]> p;
    size_t sz = D; 
    ArgGen() {
        try {
            p = nca::simd::make_aligned_unique<float>(sz);
            if (p) std::memset(p.get(), 0, sz * sizeof(float));
        } catch (...) { std::cerr << "      [FATAL] ArgGen<float*> alloc failed\n"; throw; }
    }
    void init(int seed) { if (p) { for(size_t i=0; i<sz; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; } }
    float* get() { return p.get(); }
    auto hash() { return hash_bytes(p.get(), sz * sizeof(float)); }
};
template <> struct ArgGen<const float*> {
    nca::simd::aligned_unique_ptr<float[]> p;
    size_t sz = D;
    ArgGen() {
        try {
            p = nca::simd::make_aligned_unique<float>(sz);
            if (p) std::memset(p.get(), 0, sz * sizeof(float));
        } catch (...) { std::cerr << "      [FATAL] ArgGen<const float*> alloc failed\n"; throw; }
    }
    void init(int seed) { if (p) { for(size_t i=0; i<sz; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; } }
    const float* get() { return p.get(); }
    auto hash() { return hash_bytes(p.get(), sz * sizeof(float)); }
};

template <> struct ArgGen<std::span<float>> {
    nca::simd::aligned_unique_ptr<float[]> p;
    ArgGen() : p(nca::simd::make_aligned_unique<float>(D)) { if (p) std::memset(p.get(), 0, D * sizeof(float)); }
    void init(int seed) { if (p) { for(size_t i=0; i<D; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; } }
    std::span<float> get() { return {p.get(), D}; }
    auto hash() { return hash_bytes(p.get(), D * sizeof(float)); }
};
template <> struct ArgGen<std::span<const float>> {
    nca::simd::aligned_unique_ptr<float[]> p;
    ArgGen() : p(nca::simd::make_aligned_unique<float>(D)) { if (p) std::memset(p.get(), 0, D * sizeof(float)); }
    void init(int seed) { if (p) { for(size_t i=0; i<D; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; } }
    std::span<const float> get() { return {p.get(), D}; }
    auto hash() { return hash_bytes(p.get(), D * sizeof(float)); }
};

template <> struct ArgGen<nca::linalg::MXINT8Tensor&> {
    nca::linalg::MXINT8Tensor t{D / 32};
    void init(int seed) { 
        if (t.data) std::memset(t.data, seed % 127, t.num_blocks * 32); 
        if (t.scales) std::memset(t.scales, 1, t.num_blocks); 
        if (t.w_sums) std::memset(t.w_sums, 0, t.num_blocks * 4); 
    }
    nca::linalg::MXINT8Tensor& get() { return t; }
    auto hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};
template <> struct ArgGen<const nca::linalg::MXINT8Tensor&> {
    nca::linalg::MXINT8Tensor t{D / 32};
    void init(int seed) { 
        if (t.data) std::memset(t.data, seed % 127, t.num_blocks * 32); 
        if (t.scales) std::memset(t.scales, 1, t.num_blocks); 
        if (t.w_sums) std::memset(t.w_sums, 0, t.num_blocks * 4); 
    }
    const nca::linalg::MXINT8Tensor& get() { return t; }
    auto hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};
template <> struct ArgGen<nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor t{D / 32};
    void init(int seed) { 
        if (t.data) std::memset(t.data, 128, t.num_blocks * 32); 
        if (t.scales) std::memset(t.scales, 1, t.num_blocks); 
    }
    nca::linalg::MXUINT8Tensor& get() { return t; }
    auto hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};
template <> struct ArgGen<const nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor t{D / 32};
    void init(int seed) { 
        if (t.data) std::memset(t.data, 128, t.num_blocks * 32); 
        if (t.scales) std::memset(t.scales, 1, t.num_blocks); 
    }
    const nca::linalg::MXUINT8Tensor& get() { return t; }
    auto hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};

template <> struct ArgGen<size_t> {
    void init(int) {}
    size_t get() { return D; }
    auto hash() { return (uint64_t)D; }
};
template <> struct ArgGen<float> {
    void init(int) {}
    float get() { return 1e-5f; }
    auto hash() { return 0ULL; }
};
template <> struct ArgGen<nca::layers::SSMConfig> {
    void init(int) {}
    nca::layers::SSMConfig get() { return {D, 16}; }
    auto hash() { return 0ULL; }
};
template <> struct ArgGen<nca::layers::SLAConfig> {
    void init(int) {}
    nca::layers::SLAConfig get() { return {128, 256, 256}; }
    auto hash() { return 0ULL; }
};

template <> struct ArgGen<nca::layers::HaltingState&> {
    nca::layers::HaltingState state;
    void init(int seed) { state.p_sum = 0.0f; state.remainder = 0.0f; state.steps = 0; }
    nca::layers::HaltingState& get() { return state; }
    auto hash() { return hash_bytes(&state, sizeof(state)); }
};

template <> struct ArgGen<nca::vision::ScannerConfig> {
    void init(int) {}
    nca::vision::ScannerConfig get() { return {16, 16, 32}; }
    auto hash() { return 0ULL; }
};

template <> struct ArgGen<bool&> {
    bool b = false;
    void init(int) { b = false; }
    bool& get() { return b; }
    auto hash() { return b ? 1ULL : 0ULL; }
};

template <> struct ArgGen<float&> {
    float f = 0.0f;
    void init(int) { f = 0.0f; }
    float& get() { return f; }
    auto hash() { return (uint64_t)f; }
};

template <typename Func, typename Tuple, size_t... Is>
void invoke_impl(Func f, Tuple& t, std::index_sequence<Is...>) {
    f(std::get<Is>(t).get()...);
}

template <typename Func, typename Tuple, size_t... Is>
void init_impl(Tuple& t, int seed, std::index_sequence<Is...>) {
    (std::get<Is>(t).init(seed), ...);
}

template <typename Func, typename Tuple, size_t... Is>
uint64_t hash_impl(Tuple& t, std::index_sequence<Is...>) {
    uint64_t h = 0;
    ((h ^= std::get<Is>(t).hash() + 0x9e3779b9 + (h << 6) + (h >> 2)), ...);
    return h;
}

template <typename Func> struct BenchmarkRunner;

template <typename R, typename... Args>
struct BenchmarkRunner<R(*)(Args...)> {
    static void run(const char* name, R(*f)(Args...)) {
        std::cout << "    [DEBUG] Constructing tuple...\n";
        std::tuple<ArgGen<strip_t<Args>>...> t;
        constexpr auto seq = std::index_sequence_for<Args...>{};
        
        std::cout << "    [DEBUG] Scalar check...\n";
        nca::simd::set_override_backend(nca::simd::Backend::Scalar);
        init_impl<R(*)(Args...)>(t, 42, seq);
        invoke_impl(f, t, seq);
        auto hash_scalar = hash_impl<R(*)(Args...)>(t, seq);
        
        std::cout << "    [DEBUG] AVX512 check...\n";
        nca::simd::set_override_backend(nca::simd::Backend::AVX512);
        init_impl<R(*)(Args...)>(t, 42, seq);
        invoke_impl(f, t, seq);
        auto hash_avx512 = hash_impl<R(*)(Args...)>(t, seq);
        
        if (hash_scalar == hash_avx512) {
            std::cout << "    [AMI EXACT ] Bit-perfect match.\n";
        } else {
            std::cout << "    [AMI APPROX] Outputs differ.\n";
        }
        
        nca::simd::clear_override_backend();
        
        std::cout << "    [DEBUG] Warmup...\n";
        init_impl<R(*)(Args...)>(t, 42, seq);
        invoke_impl(f, t, seq);  // Warmup
        
        std::cout << "    [DEBUG] Benching...\n";
        uint64_t runs[3];
        for (int r = 0; r < 3; ++r) {
            auto c0 = __rdtsc();
            invoke_impl(f, t, seq);
            auto c1 = __rdtsc();
            runs[r] = c1 - c0;
        }
        std::sort(std::begin(runs), std::end(runs));
        
        auto cyc = static_cast<double>(runs[1]) / D;
        std::cout << "    Cycles / Elem : " << cyc << " cycles\n\n";
    }
};

template <typename Func>
void run_benchmark(const char* name, Func f) {
    std::cout << "  [" << name << "]\n";
    BenchmarkRunner<Func>::run(name, f);
}

inline void print_hardware_info() {
    [[maybe_unused]] auto cpu = nca::simd::detect();
    std::cout << "+------------------------------------------+\n";
    std::cout << "|  NCA -- Auto-Detect Hardware Profiler    |\n";
    std::cout << "+------------------------------------------+\n\n";
    std::cout << "Active Backend: "
              << (nca::simd::best_backend() == nca::simd::Backend::AVX512 ? "AVX-512 (VNNI)" : "AVX2")
              << "\n\n";
    std::cout << "[Profiling Vector Length D=" << D << "]\n\n";
}

} // namespace nca::testing
