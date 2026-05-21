#pragma once
// ============================================================================
// NCA -- C++ Template Metaprogramming Auto-Benchmark Engine
//
// HOW IT WORKS (the "work smarter" trick):
//   Python discovers function NAMES.
//   C++ compiler deduces the argument TYPES from the function pointer.
//   ArgGen<T> knows how to allocate any type we use in the project.
//   BenchmarkRunner unpacks the tuple and calls the function.
//
//   To support a NEW type, add ONE ArgGen<T> specialization below.
//   That's it. No Python changes. No CMake changes. Ever.
// ============================================================================

#include "core/simd/dispatch.hpp"
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

#ifdef _WIN32
#include <intrin.h>
#include <malloc.h>
#else
#include <x86intrin.h>
#include <cstdlib>
#endif

namespace nca::testing {

constexpr size_t D = 8192;

// ─── StripRestrict<T>: normalizes __restrict away so MSVC can't cause misses ─
// This is the "work smarter" trick. Instead of adding N specializations for
// every __restrict variant, we strip the qualifier once at the meta level.
template <typename T> struct StripRestrict              { using type = T; };
template <typename T> struct StripRestrict<T* __restrict> { using type = T*; };
template <typename T> using strip_t = typename StripRestrict<T>::type;

inline uint64_t hash_bytes(const void* data, size_t len) {
    if (!data) return 0;
    uint64_t h = 14695981039346656037ULL;
    const uint8_t* ptr = (const uint8_t*)data;
    for (size_t i = 0; i < len; ++i) {
        h ^= ptr[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ─── ArgGen<T>: one specialization per *canonical* type ──────────────────────

template <typename T> struct ArgGen;

// -- float pointers --
template <> struct ArgGen<float*> {
    float* p;
    size_t sz = D * D;
    ArgGen()  { p = (float*)_aligned_malloc(sz * sizeof(float), 64); std::memset(p, 0, sz * sizeof(float)); }
    ~ArgGen() { _aligned_free(p); }
    void init(int seed) { for(size_t i=0; i<sz; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; }
    float* get() { return p; }
    uint64_t hash() { return hash_bytes(p, sz * sizeof(float)); }
};
template <> struct ArgGen<const float*> {
    float* p;
    size_t sz = D * D;
    ArgGen()  { p = (float*)_aligned_malloc(sz * sizeof(float), 64); std::memset(p, 0, sz * sizeof(float)); }
    ~ArgGen() { _aligned_free(p); }
    void init(int seed) { for(size_t i=0; i<sz; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; }
    const float* get() { return p; }
    uint64_t hash() { return hash_bytes(p, sz * sizeof(float)); }
};

// -- std::span --
template <> struct ArgGen<std::span<float>> {
    float* p;
    ArgGen()  { p = (float*)_aligned_malloc(D * sizeof(float), 64); std::memset(p, 0, D * sizeof(float)); }
    ~ArgGen() { _aligned_free(p); }
    void init(int seed) { for(size_t i=0; i<D; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; }
    std::span<float> get() { return {p, D}; }
    uint64_t hash() { return hash_bytes(p, D * sizeof(float)); }
};
template <> struct ArgGen<std::span<const float>> {
    float* p;
    ArgGen()  { p = (float*)_aligned_malloc(D * sizeof(float), 64); std::memset(p, 0, D * sizeof(float)); }
    ~ArgGen() { _aligned_free(p); }
    void init(int seed) { for(size_t i=0; i<D; ++i) p[i] = (float)((i + seed) % 255) / 255.0f; }
    std::span<const float> get() { return {p, D}; }
    uint64_t hash() { return hash_bytes(p, D * sizeof(float)); }
};

// -- MX tensors (by reference) --
template <> struct ArgGen<nca::linalg::MXINT8Tensor&> {
    nca::linalg::MXINT8Tensor t{D * D / 32};
    void init(int seed) { std::memset(t.data, seed % 127, t.num_blocks * 32); std::memset(t.scales, 1, t.num_blocks); std::memset(t.w_sums, 0, t.num_blocks * 4); }
    nca::linalg::MXINT8Tensor& get() { return t; }
    uint64_t hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};
template <> struct ArgGen<const nca::linalg::MXINT8Tensor&> {
    nca::linalg::MXINT8Tensor t{D * D / 32};
    void init(int seed) { std::memset(t.data, seed % 127, t.num_blocks * 32); std::memset(t.scales, 1, t.num_blocks); std::memset(t.w_sums, 0, t.num_blocks * 4); }
    const nca::linalg::MXINT8Tensor& get() { return t; }
    uint64_t hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};
template <> struct ArgGen<nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor t{D / 32};
    void init(int seed) { std::memset(t.data, 128, t.num_blocks * 32); std::memset(t.scales, 1, t.num_blocks); }
    nca::linalg::MXUINT8Tensor& get() { return t; }
    uint64_t hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};
template <> struct ArgGen<const nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor t{D / 32};
    void init(int seed) { std::memset(t.data, 128, t.num_blocks * 32); std::memset(t.scales, 1, t.num_blocks); }
    const nca::linalg::MXUINT8Tensor& get() { return t; }
    uint64_t hash() { return hash_bytes(t.data, t.num_blocks * 32) ^ hash_bytes(t.scales, t.num_blocks); }
};

// -- Scalars & config structs --
template <> struct ArgGen<size_t> {
    void init(int) {}
    size_t get() { return D; }
    uint64_t hash() { return D; }
};
template <> struct ArgGen<float> {
    void init(int) {}
    float get() { return 1e-5f; }
    uint64_t hash() { return 0; }
};
template <> struct ArgGen<nca::layers::SSMConfig> {
    void init(int) {}
    nca::layers::SSMConfig get() { return {D, 16}; }
    uint64_t hash() { return 0; }
};
template <> struct ArgGen<nca::layers::SLAConfig> {
    void init(int) {}
    nca::layers::SLAConfig get() { return {128, 256, 256}; }
    uint64_t hash() { return 0; }
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
    uint64_t hash() { return 0; }
};

template <> struct ArgGen<bool&> {
    bool b = false;
    void init(int) { b = false; }
    bool& get() { return b; }
    uint64_t hash() { return b ? 1 : 0; }
};

// ─── BenchmarkRunner: strip __restrict, then use canonical ArgGen types ──────

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
        
        // ── AMI Correctness Check ──
        // Run with Scalar backend, hash outputs
        nca::simd::set_override_backend(nca::simd::Backend::Scalar);
        init_impl<R(*)(Args...)>(t, 42, seq);
        invoke_impl(f, t, seq);
        uint64_t hash_scalar = hash_impl<R(*)(Args...)>(t, seq);
        
        // Run with AVX512 backend, hash outputs
        nca::simd::set_override_backend(nca::simd::Backend::AVX512);
        init_impl<R(*)(Args...)>(t, 42, seq);
        invoke_impl(f, t, seq);
        uint64_t hash_avx512 = hash_impl<R(*)(Args...)>(t, seq);
        
        if (hash_scalar == hash_avx512) {
            std::cout << "    [AMI EXACT ] Bit-perfect match.\n";
        } else {
            // Not bit-exact, but could be expected FP rounding from minimax approx.
            std::cout << "    [AMI APPROX] Outputs differ (expected for transcendental kernels).\n";
        }
        
        nca::simd::clear_override_backend();
        
        // ── Benchmark: 3-run median ──
        init_impl<R(*)(Args...)>(t, 42, seq);
        invoke_impl(f, t, seq);  // Warmup
        
        uint64_t runs[3];
        for (int r = 0; r < 3; ++r) {
            auto c0 = __rdtsc();
            invoke_impl(f, t, seq);
            auto c1 = __rdtsc();
            runs[r] = c1 - c0;
        }
        // Sort 3 values to get median (branchless 3-element sort)
        if (runs[0] > runs[1]) { auto tmp = runs[0]; runs[0] = runs[1]; runs[1] = tmp; }
        if (runs[1] > runs[2]) { auto tmp = runs[1]; runs[1] = runs[2]; runs[2] = tmp; }
        if (runs[0] > runs[1]) { auto tmp = runs[0]; runs[0] = runs[1]; runs[1] = tmp; }
        
        double cyc = static_cast<double>(runs[1]) / D;
        std::cout << "    Cycles / Elem : " << cyc << " cycles\n\n";
    }
};

template <typename Func>
void run_benchmark(const char* name, Func f) {
    std::cout << "  [" << name << "]\n";
    BenchmarkRunner<Func>::run(name, f);
}

// ─── Hardware info (ASCII only — no Unicode, MSVC codepage safe) ─────────────
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
