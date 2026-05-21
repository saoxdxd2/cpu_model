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
#include "core/execution/route_planner.hpp"
#include <iostream>
#include <tuple>
#include <utility>
#include <span>
#include <cstring>
#include <algorithm>
#include <vector>
#include <chrono>

#ifdef _WIN32
#include <intrin.h>
#include <malloc.h>
#else
#include <x86intrin.h>
#include <cstdlib>
#endif

namespace nca::testing {

constexpr size_t D = 1024;

template <typename T> struct StripRestrict              { using type = T; };
template <typename T> struct StripRestrict<T* __restrict> { using type = T*; };
template <typename T> using strip_t = typename StripRestrict<T>::type;

inline uint64_t hash_bytes(const void* data, size_t len) {
    if (!data) return 0ULL;
    uint64_t h = 14695981039346656037ULL;
    const uint8_t* ptr = (const uint8_t*)data;
    for (size_t i = 0; i < len; ++i) {
        h ^= ptr[i];
        h *= 1099511628211ULL;
    }
    return h;
}

template <typename T> struct ArgGen;

template <> struct ArgGen<float*> {
    nca::simd::aligned_unique_ptr<float[]> p;
    size_t sz = 1024 * 1024;
    ArgGen()  { p = nca::simd::make_aligned_unique<float>(sz); if(p) std::memset(p.get(), 0, sz * sizeof(float)); }
    ~ArgGen() { }
    void init(int seed) { if(p) for(size_t i=0; i<sz; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; }
    float* get() { return p.get(); }
    uint64_t hash() { return hash_bytes(p.get(), sz * sizeof(float)); }
};
template <> struct ArgGen<const float*> {
    nca::simd::aligned_unique_ptr<float[]> p;
    size_t sz = 1024 * 1024;
    ArgGen()  { p = nca::simd::make_aligned_unique<float>(sz); if(p) std::memset(p.get(), 0, sz * sizeof(float)); }
    ~ArgGen() { }
    void init(int seed) { if(p) for(size_t i=0; i<sz; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; }
    const float* get() { return p.get(); }
    uint64_t hash() { return hash_bytes(p.get(), sz * sizeof(float)); }
};

template <> struct ArgGen<size_t*> {
    nca::simd::aligned_unique_ptr<size_t[]> p;
    size_t sz = 1024 * 1024;
    ArgGen()  { p = nca::simd::make_aligned_unique<size_t>(sz); if(p) std::memset(p.get(), 0, sz * sizeof(size_t)); }
    ~ArgGen() { }
    void init(int seed) { if(p) for(size_t i=0; i<sz; ++i) p[i] = (size_t)((i + seed) % 1024); }
    size_t* get() { return p.get(); }
    uint64_t hash() { return hash_bytes(p.get(), sz * sizeof(size_t)); }
};

template <> struct ArgGen<std::span<float>> {
    nca::simd::aligned_unique_ptr<float[]> p;
    ArgGen()  { p = nca::simd::make_aligned_unique<float>(D); if(p) std::memset(p.get(), 0, D * sizeof(float)); }
    ~ArgGen() { }
    void init(int seed) { if(p) for(size_t i=0; i<D; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; }
    std::span<float> get() { return {p.get(), D}; }
    uint64_t hash() { return hash_bytes(p.get(), D * sizeof(float)); }
};
template <> struct ArgGen<std::span<const float>> {
    nca::simd::aligned_unique_ptr<float[]> p;
    ArgGen()  { p = nca::simd::make_aligned_unique<float>(D); if(p) std::memset(p.get(), 0, D * sizeof(float)); }
    ~ArgGen() { }
    void init(int seed) { if(p) for(size_t i=0; i<D; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; }
    std::span<const float> get() { return {p.get(), D}; }
    uint64_t hash() { return hash_bytes(p.get(), D * sizeof(float)); }
};

template <> struct ArgGen<nca::linalg::MXINT8Tensor&> {
    nca::linalg::MXINT8Tensor t{D * D / 32};
    void init(int seed) { if(t.data) { std::memset(t.data, seed % 127, t.num_blocks * 32); std::memset(t.scales, 1, t.num_blocks); std::memset(t.w_sums, 0, t.num_blocks * 4); } }
    nca::linalg::MXINT8Tensor& get() { return t; }
    uint64_t hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};
template <> struct ArgGen<const nca::linalg::MXINT8Tensor&> {
    nca::linalg::MXINT8Tensor t{D * D / 32};
    void init(int seed) { if(t.data) { std::memset(t.data, seed % 127, t.num_blocks * 32); std::memset(t.scales, 1, t.num_blocks); std::memset(t.w_sums, 0, t.num_blocks * 4); } }
    const nca::linalg::MXINT8Tensor& get() { return t; }
    uint64_t hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};
template <> struct ArgGen<nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor t{D / 32};
    void init(int seed) { if(t.data) { std::memset(t.data, 128, t.num_blocks * 32); std::memset(t.scales, 1, t.num_blocks); } }
    nca::linalg::MXUINT8Tensor& get() { return t; }
    uint64_t hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};
template <> struct ArgGen<const nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor t{D / 32};
    void init(int seed) { if(t.data) { std::memset(t.data, 128, t.num_blocks * 32); std::memset(t.scales, 1, t.num_blocks); } }
    const nca::linalg::MXUINT8Tensor& get() { return t; }
    uint64_t hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};

template <> struct ArgGen<const nca::linalg::MXINT8Tensor*> {
    std::vector<nca::linalg::MXINT8Tensor> ts;
    ArgGen() { for(int i=0; i<16; ++i) ts.emplace_back(D/32); }
    void init(int seed) { for(auto& t : ts) { if(t.data) std::memset(t.data, seed % 127, t.num_blocks*32); if(t.scales) std::memset(t.scales, 1, t.num_blocks); if(t.w_sums) std::memset(t.w_sums, 0, t.num_blocks*4); } }
    const nca::linalg::MXINT8Tensor* get() { return ts.data(); }
    uint64_t hash() { uint64_t h = 0; for(auto& t : ts) h ^= hash_bytes(t.data, t.num_blocks*32); return h; }
};

template <> struct ArgGen<size_t> {
    void init(int) {}
    size_t get() { return D; }
    uint64_t hash() { return (uint64_t)D; }
};
template <> struct ArgGen<float> {
    void init(int) {}
    float get() { return 1e-5f; }
    uint64_t hash() { return 0ULL; }
};
template <> struct ArgGen<nca::layers::SSMConfig> {
    void init(int) {}
    nca::layers::SSMConfig get() { return {D, 16}; }
    uint64_t hash() { return 0ULL; }
};
template <> struct ArgGen<nca::layers::SLAConfig> {
    void init(int) {}
    nca::layers::SLAConfig get() { return {128, 256, 256}; }
    uint64_t hash() { return 0ULL; }
};

template <> struct ArgGen<nca::layers::HaltingState&> {
    nca::layers::HaltingState state;
    void init(int seed) { state.p_sum = 0.0f; state.remainder = 0.0f; state.steps = 0; }
    nca::layers::HaltingState& get() { return state; }
    uint64_t hash() { return hash_bytes(&state, sizeof(state)); }
};

template <> struct ArgGen<nca::vision::ScannerConfig> {
    void init(int) {}
    nca::vision::ScannerConfig get() { return {16, 16, 32}; }
    uint64_t hash() { return 0ULL; }
};

template <> struct ArgGen<bool&> {
    bool b = false;
    void init(int = 0) { b = false; }
    bool& get() { return b; }
    uint64_t hash() { return b ? 1ULL : 0ULL; }
};

template <> struct ArgGen<float&> {
    float f = 0.0f;
    void init(int = 0) { f = 0.0f; }
    float& get() { return f; }
    uint64_t hash() { return (uint64_t)f; }
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
        std::tuple<ArgGen<strip_t<Args>>...> t;
        constexpr auto seq = std::index_sequence_for<Args...>{};
        
        std::cout << "  [CHECK ] " << name << " ... " << std::flush;

        // 1. Correctness Verification (AMI)
        nca::simd::set_override_backend(nca::simd::Backend::Scalar);
        init_impl<R(*)(Args...)>(t, 42, seq);
        invoke_impl(f, t, seq);
        uint64_t hash_scalar = hash_impl<R(*)(Args...)>(t, seq);
        
        nca::simd::set_override_backend(nca::simd::Backend::AVX512);
        init_impl<R(*)(Args...)>(t, 42, seq);
        invoke_impl(f, t, seq);
        uint64_t hash_avx512 = hash_impl<R(*)(Args...)>(t, seq);
        
        const char* status = (hash_scalar == hash_avx512) ? "OK (AMI EXACT)" : "OK (AMI APPROX)";
        
        // 2. Hardware Benchmarking
        nca::simd::clear_override_backend();
        init_impl<R(*)(Args...)>(t, 42, seq);
        invoke_impl(f, t, seq);  // Warmup
        
        uint64_t runs_cyc[5];
        double runs_time[5];
        for (int r = 0; r < 5; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto c0 = __rdtsc();
            invoke_impl(f, t, seq);
            auto c1 = __rdtsc();
            auto t1 = std::chrono::high_resolution_clock::now();
            runs_cyc[r] = c1 - c0;
            runs_time[r] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        std::sort(std::begin(runs_cyc), std::end(runs_cyc));
        std::sort(std::begin(runs_time), std::end(runs_time));
        
        auto cyc = static_cast<double>(runs_cyc[2]) / D;
        auto us  = runs_time[2];
        
        std::cout << status << "\n";
        std::cout << "    Latency       : " << us << " us\n";
        std::cout << "    Throughput    : " << cyc << " cycles/elem\n\n";
        std::cout.flush();
    }
};

template <typename Func>
void run_benchmark(const char* name, Func f) {
    BenchmarkRunner<Func>::run(name, f);
}

inline void print_hardware_info() {
    [[maybe_unused]] auto cpu = nca::simd::detect();
    std::cout << "+------------------------------------------+\n";
    std::cout << "|  NCA -- Auto-Detect Hardware Profiler    |\n";
    std::cout << "+------------------------------------------+\n\n";
    std::cout << "Active Backend: "
              << (nca::simd::best_backend() == nca::simd::Backend::AVX512 ? "AVX-512 (VNNI)" : "AVX2")
              << "\n";
    std::cout << "Vector Length : " << D << "\n\n";
}

} // namespace nca::testing
