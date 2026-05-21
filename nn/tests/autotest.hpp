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

// ─── ArgGen<T>: one specialization per *canonical* type ──────────────────────

template <typename T> struct ArgGen;

// -- float pointers --
template <> struct ArgGen<float*> {
    float* p;
    ArgGen()  { p = (float*)_aligned_malloc(D * D * sizeof(float), 64); std::memset(p, 0, D * D * sizeof(float)); }
    ~ArgGen() { _aligned_free(p); }
    float* get() { return p; }
};
template <> struct ArgGen<const float*> {
    float* p;
    ArgGen()  { p = (float*)_aligned_malloc(D * D * sizeof(float), 64); std::memset(p, 0, D * D * sizeof(float)); }
    ~ArgGen() { _aligned_free(p); }
    const float* get() { return p; }
};

// -- std::span --
template <> struct ArgGen<std::span<float>> {
    float* p;
    ArgGen()  { p = (float*)_aligned_malloc(D * sizeof(float), 64); std::memset(p, 0, D * sizeof(float)); }
    ~ArgGen() { _aligned_free(p); }
    std::span<float> get() { return {p, D}; }
};
template <> struct ArgGen<std::span<const float>> {
    float* p;
    ArgGen()  { p = (float*)_aligned_malloc(D * sizeof(float), 64); std::memset(p, 0, D * sizeof(float)); }
    ~ArgGen() { _aligned_free(p); }
    std::span<const float> get() { return {p, D}; }
};

// -- MX tensors (by reference) --
template <> struct ArgGen<nca::linalg::MXINT8Tensor&> {
    nca::linalg::MXINT8Tensor t{D * D / 32};
    nca::linalg::MXINT8Tensor& get() { return t; }
};
template <> struct ArgGen<const nca::linalg::MXINT8Tensor&> {
    nca::linalg::MXINT8Tensor t{D * D / 32};
    const nca::linalg::MXINT8Tensor& get() { return t; }
};
template <> struct ArgGen<nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor t{D / 32};
    nca::linalg::MXUINT8Tensor& get() { return t; }
};
template <> struct ArgGen<const nca::linalg::MXUINT8Tensor&> {
    nca::linalg::MXUINT8Tensor t{D / 32};
    const nca::linalg::MXUINT8Tensor& get() { return t; }
};

// -- Scalars & config structs --
template <> struct ArgGen<size_t> {
    size_t get() { return D; }
};
template <> struct ArgGen<float> {
    float get() { return 1e-5f; }
};
template <> struct ArgGen<nca::layers::SSMConfig> {
    nca::layers::SSMConfig get() { return {D, 16}; }
};
template <> struct ArgGen<nca::layers::SLAConfig> {
    nca::layers::SLAConfig get() { return {128, 256, 256}; }
};

template <> struct ArgGen<nca::layers::HaltingState&> {
    nca::layers::HaltingState state;
    nca::layers::HaltingState& get() { return state; }
};

template <> struct ArgGen<nca::vision::ScannerConfig> {
    nca::vision::ScannerConfig get() { return {16, 16, 32}; } // 16x16x32 = 8192 elements. Prevents Out-Of-Bounds on ArgGen buffers.
};

template <> struct ArgGen<bool&> {
    bool b = false;
    bool& get() { return b; }
};

// ─── BenchmarkRunner: strip __restrict, then use canonical ArgGen types ──────

template <typename Func, typename Tuple, size_t... Is>
void invoke_impl(Func f, Tuple& t, std::index_sequence<Is...>) {
    f(std::get<Is>(t).get()...);
}

template <typename Func> struct BenchmarkRunner;

template <typename R, typename... Args>
struct BenchmarkRunner<R(*)(Args...)> {
    static void run(const char* name, R(*f)(Args...)) {
        std::tuple<ArgGen<strip_t<Args>>...> t;
        constexpr auto seq = std::index_sequence_for<Args...>{};
        // Warmup
        invoke_impl(f, t, seq);
        // Timed
        auto c0 = __rdtsc();
        invoke_impl(f, t, seq);
        auto c1 = __rdtsc();
        double cyc = static_cast<double>(c1 - c0) / D;
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
